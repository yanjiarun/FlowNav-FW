import argparse
from core.flow_matching import train_flow_matching

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data/data_flow_nav/episode_1", help="Directory for training data")
    parser.add_argument("--save_dir", type=str, default="./models/pre_train_resnet_unet_224reshape_init_gaussion200", help="Directory to save the model")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument("--return_waypoints_id", action="store_true", default=False, help="Return waypoints id")
    parser.add_argument("--seq_length", type=int, default=1, help="Sequence length")
    parser.add_argument("--future_steps", type=int, default=16, help="Future steps")
    args = parser.parse_args()
    train_flow_matching(args)