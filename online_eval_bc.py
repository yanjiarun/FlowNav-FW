import argparse
from core.online_nav_bc import OnlineNavigatorBC, CAMERA_CONFIG_DEFAULT
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
    
    parser = argparse.ArgumentParser(description='Validate BC Policy for UAV Navigation')
    parser.add_argument('--save_folder', type=str, default='./data_flow_nav_bc130_save_images', help='Directory to save validation results')
    parser.add_argument('--model_path', type=str, default="./models/bc_models/bc_checkpoint_epoch_130.pth", help='Path to trained BC model')
    parser.add_argument('--waypoints', type=str, default=waypoints, help='Path to waypoints file')
    parser.add_argument('--camera_config', type=str, default=CAMERA_CONFIG_DEFAULT, help='Camera configuration')
    parser.add_argument('--episode_num', type=int, default=1, help='Number of validation episodes')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--seq_length', type=int, default=1, help='Input sequence length (history frames)')
    parser.add_argument('--pred_horizon', type=int, default=16, help='Prediction horizon (future steps)')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--action_dim', type=int, default=2, help='Action dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--use_lstm', type=bool, default=True, help='Whether to use LSTM for temporal processing')
    args = parser.parse_args()
    
    # Create online navigator and run
    online_nav = OnlineNavigatorBC(args)
    online_nav.run()
