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
from core.diffusion_policy import DiffusionPolicy  # Import correct diffusion policy class
from collections import deque
from torchvision import transforms
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

class OnlineNavigatorDP:
    def __init__(self, args):
        self.save_folder = args.save_folder
        self.model_path = args.model_path
        self.waypoints = args.waypoints
        self.camera_config = args.camera_config
        self.episode_num = args.episode_num
        self.debug = args.debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_length = args.seq_length
        self.pred_horizon = args.pred_horizon

        # Load diffusion policy model
        self.nav_policy = DiffusionPolicy(
            history_length=self.seq_length,
            pred_horizon=self.pred_horizon
        ).to(self.device)
        checkpoint = torch.load(self.model_path)
        if 'model_state_dict' in checkpoint:
            self.nav_policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.nav_policy.load_state_dict(checkpoint)
        self.nav_policy.eval()

        self.initialize_unity_bridge()

    def initialize_unity_bridge(self):
        """
        Initialize Unity bridge
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
        Run validation process
        """
        for episode_num in range(1, self.episode_num + 1):
            self.run_one_episode(episode_num)
        self.unity_bridge.close()

    def run_one_episode(self, episode_num):
        """
        Run one validation episode
        """
        save_episode_folder = os.path.join(self.save_folder, f"episode_{episode_num}")
        save_image_folder = os.path.join(save_episode_folder, "imgs")
        os.makedirs(save_image_folder, exist_ok=True)
        
        # Data recording
        save_positions = []
        save_quaternions = []
        save_commands = []
        save_waypoint_indices = []
        save_lla_data = []
        save_vned_data = []
        save_vair_data = []
        save_atti_data = []

        print(f"\n{'='*50}")
        print(f"Starting Validation Episode {episode_num}")
        print(f"{'='*50}\n")

        frame_id = 1
        simulation_dt = 0.05
        self.current_waypoint_idx = 0
        self.current_waypoint = self.waypoints[self.current_waypoint_idx]
        self.mission_complete = False

        # Create fixed-wing aircraft and controller
        fw, controller = self.create_fw_and_controller(init_position=self.waypoints[0].tolist())

        # Initialize image history queue
        image_history = deque(maxlen=self.seq_length)
        action_buffer = deque(maxlen=self.pred_horizon)
        current_action_index = 0
        
        # Define preprocessing transforms same as during training
        resize_shape = (256, 256)  # Consistent with training
        crop_shape = (224, 224)    # Consistent with training

        # Create preprocessing transforms
        preprocess = transforms.Compose([
            transforms.Resize(resize_shape),  # Resize
            transforms.CenterCrop(crop_shape)  # Center crop
        ])

        try:
            while not self.mission_complete:
                current_pos = fw.get_state_trans().get_position()
                current_quaternion = fw.get_state_trans().get_quaternion()
                
                # Update waypoint
                self.update_waypoint(current_pos)
                
                # Get current image
                image = self.get_current_image(frame_id, current_pos, current_quaternion)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Add to image history queue
                image_history.append(image)
                
                # If insufficient history images, pad with current image
                while len(image_history) < self.seq_length:
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
                
                # If need to generate new action sequence
                if current_action_index >= len(action_buffer):
                    with torch.no_grad():
                        # Use diffusion model's predict_action method to sample action sequence
                        action_sequence = self.nav_policy.predict_action(
                            image_sequence,
                            num_inference_steps=100
                        ).squeeze(0).cpu().numpy()  # [T, Da]
                        
                        # Reset action buffer
                        action_buffer = deque(action_sequence)
                        current_action_index = 0
                        print(f"Generated new action sequence: {action_sequence}")
                
                # Get current action
                current_action = action_buffer[current_action_index]
                current_action_index += 1
                
                # Convert to heading angle
                command = self.vector_to_heading(current_action)
                print(f"Frame {frame_id}: Command = {command:.2f}°")
                
                # Send control command
                control = controller.heading_and_altitude_hold(18.0, command, 100.0, fw.get_state())
                
                # Record data
                lla = fw.get_state().get_position()
                vned = fw.get_state().get_velocity()
                vair = fw.get_state().get_airspeed()
                atti = fw.get_state().get_attitude()
                
                save_positions.append(current_pos)
                save_quaternions.append(current_quaternion)
                save_commands.append(command)
                save_waypoint_indices.append(self.current_waypoint_idx)
                save_lla_data.append(lla)
                save_vned_data.append(vned)
                save_vair_data.append(vair)
                save_atti_data.append(atti)
                
                # Save current image
                self.save_image(image, frame_id, save_image_folder)
                
                # Update UAV state
                fw.run(simulation_dt, control)
                frame_id += 1

                if frame_id > 5400:  # Maximum frame limit to prevent infinite loop
                    print("Reached maximum frame limit, ending episode.")
                    break
                
                # Add delay to match real-time
                # time.sleep(simulation_dt)

        except KeyboardInterrupt:
            print("Episode interrupted by user")
        finally:
            # Save data
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
        Update current waypoint based on UAV position
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

    def save_image(self, image, frame_id, save_folder):
        """Save current image"""
        rgb_file_path = os.path.join(save_folder, f"image_{frame_id:04d}.png")
        try:
            cv2.imwrite(rgb_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Failed to save image to {rgb_file_path}: {e}")

    def get_current_image(self, frame_id, current_pos, current_quaternion):
        """Get current image from Unity"""
        position_unity = convert_position_to_unity(current_pos)
        quaternion_unity = convert_quaternion_to_unity(current_quaternion)
        unity_pose_kwargs = {
            'frame_id': frame_id,
            'vehicle_id': 'FPV',
            'position': position_unity,
            'rotation': quaternion_unity,
            'cameras': self.camera_config
        }
        
        # Initial frame needs multiple pose sends to ensure Unity receives it
        if frame_id == 1:
            for _ in range(10):
                self.unity_bridge.send_pose(**unity_pose_kwargs)
                time.sleep(0.01)
        
        # Continuously send pose until matching image is obtained
        while True:
            self.unity_bridge.send_pose(**unity_pose_kwargs)
            time.sleep(0.033)  # ~30 FPS
            img_data = self.unity_bridge.get_latest_image(0, 0)
            if img_data is not None and img_data["frame_id"] == frame_id:
                rgb_img = cv2.cvtColor(img_data["rgb"], cv2.COLOR_RGB2BGR)
                if self.debug:
                    print(f"Image captured for frame {frame_id}")
                return rgb_img

    @staticmethod
    def vector_to_heading(vector):
        """
        Convert 2D vector to heading angle (0-360 degrees)
        """
        x, y = vector
        heading_rad = math.atan2(y, x)
        heading_deg = math.degrees(heading_rad) % 360
        return heading_deg

    @staticmethod
    def create_fw_and_controller(init_position=[30, -120, 100]):
        """
        Create fixed-wing aircraft and controller instances
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