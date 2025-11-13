import argparse
from core.online_nav_diffusion_policy import OnlineNavigatorDP, CAMERA_CONFIG_DEFAULT
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
    
    parser = argparse.ArgumentParser(description="Validate Diffusion Policy for UAV Navigation")
    parser.add_argument("--save_folder", type=str, default="./data_flow_nav_diffusion_policy_diffusers_scheduler200_save_images1", help="Directory to save validation data")
    parser.add_argument("--model_path", type=str, default="./models/diffusion_models_diffusers_scheduler/diffusion_checkpoint_epoch_200.pth", help="Path to the trained model")
    parser.add_argument("--waypoints", type=str, default=waypoints, help="Waypoints array")
    parser.add_argument("--camera_config", type=str, default=CAMERA_CONFIG_DEFAULT, help="Camera configuration")
    parser.add_argument("--episode_num", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seq_length", type=int, default=1, help="Input sequence length")
    parser.add_argument("--pred_horizon", type=int, default=16, help="Prediction horizon")
    args = parser.parse_args()

    # Create online navigator and run
    online_nav = OnlineNavigatorDP(args)
    online_nav.run()
