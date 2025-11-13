import argparse
from core.online_nav import OnlineNavigator, CAMERA_CONFIG_DEFAULT
import numpy as np

if __name__ == "__main__":
    # Define waypoint sequence (ENU coordinate system)
    waypoints = np.array([
        [30, -120, 100],
        [-400, 250, 100],
        [-100, 600, 100],
        [-200, 1200, 100],
        [300, 1500, 100],
        [1000, 2000, 100],
        [1500, 1600, 100],
        [2100, 1000, 100]  # Final waypoint
    ])
    
    parser = argparse.ArgumentParser(description="Online Evaluation of Flow Matching Model")
    parser.add_argument("--save_folder", type=str, default="./data_flow_nav_pre_train_resnet_unet_224reshape200_save_images", help="Directory to save the model")
    parser.add_argument("--model_path", type=str, default="./models/pre_train_resnet_unet_224reshape200/flow_matching_policy_final.pth", help="Path to the trained model")
    parser.add_argument("--waypoints", type=str, default=waypoints, help="Path to the waypoints file")
    parser.add_argument("--camera_config", type=str, default=CAMERA_CONFIG_DEFAULT, help="Camera configuration")
    parser.add_argument("--episode_num", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Create online navigator and run
    online_nav = OnlineNavigator(args)
    online_nav.run()
