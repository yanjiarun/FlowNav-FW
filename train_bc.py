import argparse
from core.bc import train_bc_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BC Policy for UAV Navigation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument("--data_dir", type=str, default="./data/data_flow_nav/episode_1", help="Directory for training data")
    parser.add_argument('--save_dir', type=str, default='./models/bc_models', help='Directory to save models')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--seq_length', type=int, default=1, help='Input sequence length (history frames)')
    parser.add_argument('--future_steps', type=int, default=16, help='Prediction horizon (future steps)')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--action_dim', type=int, default=2, help='Action dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--use_lstm', type=bool, default=True, help='Whether to use LSTM for temporal processing')
    args = parser.parse_args()
    train_bc_policy(args)