import os
import math
import time
from pathlib import Path
import cv2
import numpy as np
import jsbsim
from AFPV_Kit.objects.fixedwing import FixedWing
from AFPV_Kit.utils.autopilot import X8Autopilot
from AFPV_Kit.bridges.unity_bridge import UnityBridge
from AFPV_Kit.common.transfer import convert_position_to_unity, convert_quaternion_to_unity, calculate_camera_matrix
import torch
from core.flow_matching import FlowMatchingPolicy, sample_action
from collections import deque
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

CAMERA_CONFIG_DEFAULT = [{
    "ID": "Camera",
    "width": 1024,
    "height": 768,
    "channels": 3,
    "fov": 90.0,
    "nearClipPlane": [0.5, 0.5, 0.5],
    "farClipPlane": [1000.0, 550.0, 1000.0],
    "enabledLayers": [False, False],
    "T_BC": calculate_camera_matrix(pitch_deg=-15)
}]

class OnlineNavigator:
    def __init__(self, args):
        self.save_folder = args.save_folder
        self.model_path = args.model_path
        self.waypoints = args.waypoints
        self.camera_config = args.camera_config
        self.episode_num = args.episode_num
        self.debug = args.debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nav_policy = FlowMatchingPolicy().to(self.device)
        checkpoint = torch.load(self.model_path)
        if 'model_state_dict' in checkpoint:
            self.nav_policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.nav_policy.load_state_dict(checkpoint)
        self.nav_policy.eval()

        self.initialize_unity_bridge()

    def initialize_unity_bridge(self):
        """
        Initialize the Unity bridge for communication with the simulation.
        """
        vehicles_info = {"vehicles": [{
            "ID": "FPV",
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0, 0, 1],
            "scale": [0.05, 0.05, 0.05],
            "cameras": self.camera_config
        }]}
        self.unity_bridge = UnityBridge(vehicles_info=vehicles_info)
        self.unity_bridge.start()

    def run(self):
        """
        Run the data collection process.
        """
        for episode_num in range(1, self.episode_num + 1):
            self.run_one_episode_with_save(episode_num)
        self.unity_bridge.close()

    def run_one_episode_with_save(self, episode_num):
        """
        Run one episode of the simulation.
        """
        save_episode_folder = os.path.join(self.save_folder, f"episode_{episode_num}")
        save_image_folder = os.path.join(save_episode_folder, "imgs")
        os.makedirs(save_image_folder, exist_ok=True)
        save_image_flag = True
        save_positions, save_quaternions = [], []
        save_commands, save_waypoint_indices = [], []
        save_lla_data, save_vned_data, save_vair_data, save_atti_data = [], [], [], []

        print(f"\n{'='*50}")
        print(f"Starting Episode {episode_num}")
        print(f"{'='*50}\n")

        frame_id = 1
        simulation_dt = 0.05
        self.current_waypoint_idx = 0
        self.current_waypoint = self.waypoints[self.current_waypoint_idx]
        self.mission_complete = False

        fw, controller = self.create_fw_and_controller(init_position=self.waypoints[0].tolist())

        # Initialize image history queue
        history_length = 1
        image_history = deque(maxlen=history_length)
        predicted_vector_seq = None
        current_vector_index = 0
        
        # Define preprocessing transforms same as during training
        resize_shape = (256, 256)  # Consistent with training
        crop_shape = (224, 224)    # Consistent with training

        # Create preprocessing transforms
        preprocess = transforms.Compose([
            transforms.Resize(resize_shape),  # Resize
            transforms.CenterCrop(crop_shape)  # Center crop
        ])

        try:
            while True:
                current_pos = fw.get_state_trans().get_position()
                current_quaternion = fw.get_state_trans().get_quaternion()
                
                self.update_waypoint(current_pos)

                # If mission complete, break
                if self.mission_complete:
                    break

                image = self.get_current_image(frame_id, current_pos, current_quaternion)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image_history.append(image) 
                # If insufficient history images, pad with current image
                while len(image_history) < history_length:
                    image_history.append(image)

                # Create image sequence tensor
                image_sequence = []
                for img in image_history:
                    img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
                    img_pil = preprocess(img_pil)  # Apply preprocessing
                    img_tensor = torch.FloatTensor(np.array(img_pil)).permute(2, 0, 1) / 255.0  # [C, H, W], 0-1
                    image_sequence.append(img_tensor)
                
                # Stack into sequence [seq_length, C, H, W]
                image_sequence = torch.stack(image_sequence).unsqueeze(0).to(self.device)  # [1, seq_length, C, H, W]

                if predicted_vector_seq is None or current_vector_index >= len(predicted_vector_seq):
                    with torch.no_grad():
                        predicted_vector_seq = sample_action(self.nav_policy, image_sequence, num_steps=50, device=self.device)
                        predicted_vector_seq = predicted_vector_seq.cpu().numpy().squeeze()
                        current_vector_index = 0
                        # print(f"Predicted vector sequence: {predicted_vector_seq}")
                        # vector_seq_to_heading = self.vector_seq_to_heading(predicted_vector_seq)
                        # print(f"Predicted heading sequence: {vector_seq_to_heading}")

                command = self.vector_to_heading(predicted_vector_seq[current_vector_index])
                current_vector_index += 1
                print("command:", command)

                control = controller.heading_and_altitude_hold(18.0, command, 100.0, fw.get_state())

                lla = fw.get_state().get_position()
                vned = fw.get_state().get_velocity()
                vair = fw.get_state().get_airspeed()
                atti = fw.get_state().get_attitude()

                if self.debug:
                    print('='*50)
                    print(f'Position ENU: {current_pos}')
                    print(f'Target waypoint: {self.current_waypoint}, Command: {command:.2f}°')
                    print(f'LLA: {lla[0]:.6f}, {lla[1]:.6f}, {lla[2]:.2f}')
                    print(f'UAV Ground Speed: {vned[0]:.2f}, {vned[1]:.2f}, {vned[2]:.2f}, UAV Airspeed: {vair:.2f}')
                    print(f'Attitude: roll={math.degrees(atti[0]):.2f}°, pitch={math.degrees(atti[1]):.2f}°, yaw={math.degrees(atti[2]):.2f}°')   
                    print('='*50)

                save_positions.append(current_pos)
                save_quaternions.append(current_quaternion)
                save_commands.append(command)
                save_waypoint_indices.append(self.current_waypoint_idx)
                save_lla_data.append(lla)
                save_vned_data.append(vned)
                save_vair_data.append(vair)
                save_atti_data.append(atti)

                self.get_and_save_imgs(frame_id, current_pos, current_quaternion, save_image_flag, save_image_folder)

                fw.run(simulation_dt, control)
                frame_id += 1

                if frame_id > 5400:  # Maximum frame limit to prevent infinite loop
                    print("Reached maximum frame limit, ending episode.")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            np.save(os.path.join(save_episode_folder, "positions.npy"), np.array(save_positions))
            np.save(os.path.join(save_episode_folder, "quaternions.npy"), np.array(save_quaternions))
            np.save(os.path.join(save_episode_folder, "commands.npy"), np.array(save_commands))
            np.save(os.path.join(save_episode_folder, "waypoint_indices.npy"), np.array(save_waypoint_indices))
            np.save(os.path.join(save_episode_folder, "waypoints.npy"), self.waypoints)
            np.save(os.path.join(save_episode_folder, "lla_data.npy"), np.array(save_lla_data))
            np.save(os.path.join(save_episode_folder, "vned_data.npy"), np.array(save_vned_data))
            np.save(os.path.join(save_episode_folder, "vair_data.npy"), np.array(save_vair_data))
            np.save(os.path.join(save_episode_folder, "atti_data.npy"), np.array(save_atti_data))
            
            print(f"Episode {episode_num} data saved to {save_episode_folder}")

    def update_waypoint(self, current_pos):
        """
        Update the current waypoint index based on the UAV's position.
        """
        distance = np.linalg.norm(np.array(current_pos) - np.array(self.current_waypoint))
        if distance < 10.0:
            if self.current_waypoint_idx == len(self.waypoints) - 1:
                self.mission_complete = True
                print("Mission complete! Reached final waypoint.")
            else:
                self.current_waypoint_idx += 1
                self.current_waypoint = self.waypoints[self.current_waypoint_idx]
                print(f"Reached waypoint {self.current_waypoint_idx}, moving to next")

    def get_and_save_imgs(self, frame_id, current_pos, current_quaternion, save_image_flag, save_image_folder):
        """
        Get images from Unity and save them if required.
        """
        position_unity = convert_position_to_unity(current_pos)
        quaternion_unity = convert_quaternion_to_unity(current_quaternion)
        unity_pose_kwargs = {
            'frame_id': frame_id,
            'vehicle_id': 'FPV',
            'position': position_unity,
            'rotation': quaternion_unity,
            'cameras': self.camera_config
        }
        if frame_id == 1:
            for _ in range(100):
                self.unity_bridge.send_pose(**unity_pose_kwargs)
                time.sleep(0.01)
            self.debug and print(f"Send frame_id: {frame_id}")
        while True:
            self.unity_bridge.send_pose(**unity_pose_kwargs)
            time.sleep(0.1)
            img_data = self.unity_bridge.get_latest_image(0,0)
            if (img_data is not None) and (img_data["frame_id"] == frame_id):
                if save_image_flag:
                    rgb_img = cv2.cvtColor(img_data["rgb"], cv2.COLOR_RGB2BGR)
                    rgb_file_path = f"{save_image_folder}/image_{frame_id:04d}.png"
                    try:
                        cv2.imwrite(rgb_file_path, rgb_img)                                        
                    except Exception as e:
                        print(f"Failed to save image to {rgb_file_path}: {e}")
                self.debug and print(f"Image saved to {rgb_file_path}\n Frame ID: {frame_id}, Timestamp: {img_data['timestamp']:.6f}")
                break

    def get_current_image(self, frame_id, current_pos, current_quaternion):
        """
        Get current image from Unity.
        """
        position_unity = convert_position_to_unity(current_pos)
        quaternion_unity = convert_quaternion_to_unity(current_quaternion)
        unity_pose_kwargs = {
            'frame_id': frame_id,
            'vehicle_id': 'FPV',
            'position': position_unity,
            'rotation': quaternion_unity,
            'cameras': self.camera_config
        }
        if frame_id == 1:
            for _ in range(10):
                self.unity_bridge.send_pose(**unity_pose_kwargs)
                time.sleep(0.01)
        while True:
            self.unity_bridge.send_pose(**unity_pose_kwargs)
            time.sleep(0.033)  # ~30 FPS
            img_data = self.unity_bridge.get_latest_image(0, 0)
            if img_data is not None:
                if img_data["frame_id"] == frame_id:
                    rgb_img = cv2.cvtColor(img_data["rgb"], cv2.COLOR_RGB2BGR)
                    if self.debug:
                        print(f"Image captured for frame {frame_id}")
                    return rgb_img

    @staticmethod
    def vector_to_heading(vector):
        """
        Convert a 2D vector (x, y) to a heading angle (degrees).
        """
        x, y = vector
        heading = math.degrees(math.atan2(y, x)) % 360
        return heading
    
    @staticmethod
    def vector_seq_to_heading(vector_seq):
        """
        Convert a sequence of 2D vectors to heading angles (degrees).
        """
        headings = []
        for vector in vector_seq:
            x, y = vector
            heading = math.degrees(math.atan2(y, x)) % 360
            headings.append(heading)
        return headings

    @staticmethod
    def calculate_desired_heading(current_pos, target_pos):
        """
        Calculate the desired heading angle from the current position to the target position.
        :param current_pos: Current position [x, y, z]
        :param target_pos: Target position [x, y, z]
        :return: Desired heading angle (degrees), range 0 to 360
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        heading_deg = math.degrees(math.atan2(dx, dy)) % 360
        return heading_deg

    @staticmethod
    def create_fw_and_controller(init_position=[30, -120, 100]):
        """
        Create FixedWing and Controller instances.
        """
        jsbsim_config = {
            "jsbsim_root": str(Path(jsbsim.__file__).parent),
            "dt": 0.001,
            "aircraft": "x8",
            "init_conditions": {
                "lat_deg": 35.6541666666666,
                "lon_deg": 139.75625,
                "alt_ft": init_position[2] / 0.3048,
                "airspeed_fps": 18 / 0.3048
            }
        }
        fw = FixedWing(init_position=init_position[:2]+[0], jsb_config=jsbsim_config)
        controller = X8Autopilot()
        return fw, controller