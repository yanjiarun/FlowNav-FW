import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import math
from datetime import datetime
from core.datasets import UAVNavDataset
from torchvision.models import (
    squeezenet1_1, SqueezeNet1_1_Weights
)
import time
from core.unet import ConditionalUnet1D
from core.resnet import get_resnet
from core.resnet import replace_bn_with_gn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowMatchingPolicy(nn.Module):
    """
    Flow Matching based UAV navigation policy network (multi-history image input)
    Input: image sequence (k historical images)
    Output: 2D heading vector [x, y], range [-1, 1]
    """
    def __init__(self, image_size=128, action_dim=2, hidden_dim=128, history_length=1, pred_horizon=16):
        super().__init__()
        self.action_dim = action_dim
        self.history_length = history_length  # Number of historical images
        self.pred_horizon = pred_horizon  # Predicted action sequence length
        
        # Use SqueezeNet as visual encoder
        self.visual_encoder = get_resnet('resnet18')  # Each image outputs 512-dim features
        self.visual_encoder = replace_bn_with_gn(self.visual_encoder)

        # Flow Matching network
        vision_feature_dim = 512 * history_length  # ResNet18 output dimension
        self.flow_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=vision_feature_dim)

    def forward(self, image_sequence, action_noise, t):
        """
        Args:
            image_sequence: input image sequence [B, history_length, C, H, W]
            action_noise: noise action [B, action_dim], range [-1, 1]
            t: timestep [B, 1]
        
        Returns:
            predicted_flow: predicted velocity field [B, action_dim], range [-1, 1]
        """
        sequence_features = self.visual_encoder(image_sequence.flatten(end_dim=1))  # [B * history_length, 512]
        sequence_features = sequence_features.reshape(*image_sequence.shape[:2], -1)  # [B, history_length, 512]
        obs_cond = sequence_features.flatten(start_dim=1)  # [B, history_length * 512]

        # print(f"action_noise shape: {action_noise.shape}, t shape: {t.shape}, obs_cond shape: {obs_cond.shape}")
        # Predict velocity field
        predicted_flow = self.flow_net(action_noise, t, obs_cond)  # [B, action_dim]
        
        return predicted_flow


def flow_matching_loss(model, images, target_actions, device):
    """
    Flow Matching loss function
    """
    batch_size = images.shape[0]
    pred_horizon = target_actions.shape[1]  # Get target action sequence length
    
    # Randomly sample timesteps
    t = torch.rand(batch_size, 1, device=device)
    
    # Sample noise action from prior distribution (uniform distribution [-1, 1])
    # Ensure noise action has same shape as target action sequence
    action_noise = torch.rand(batch_size, pred_horizon, model.action_dim, device=device) * 2 - 1
    
    # action_noise = torch.randn(batch_size, model.pred_horizon, model.action_dim, device=device)

    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    # Here x_0 is noise action sequence, x_1 is target action sequence
    interpolated_actions = (1 - t.unsqueeze(-1)) * action_noise + t.unsqueeze(-1) * target_actions
    
    # True velocity field: u_t = x_1 - x_0
    true_flow = target_actions - action_noise
    
    # Model predicted velocity field
    predicted_flow = model(images, interpolated_actions, t)
    
    # Ensure predicted flow field has same shape as true flow field
    if predicted_flow.shape[1] != pred_horizon:
        raise ValueError(f"Model output sequence length ({predicted_flow.shape[1]}) "
                         f"does not match target sequence length ({pred_horizon})")
    
    # Calculate MSE loss
    loss = torch.mean((predicted_flow - true_flow) ** 2)
    
    return loss

def sample_action(model, images, num_steps=50, device="cpu"):
    """
    Sample action from learned flow
    """
    batch_size = images.shape[0]
    
    # Sample initial noise from prior distribution (uniform distribution [-1, 1])
    actions = torch.rand(batch_size, model.pred_horizon, model.action_dim, device=device) * 2 - 1  # Range [-1, 1]
    
    # actions = torch.randn(batch_size, model.pred_horizon, model.action_dim, device=device)

    # Use Euler method to solve ODE
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch_size,), step / num_steps, device=device)
        flow = model(images, actions, t)
        actions = actions + flow * dt
        
        # # Ensure actions stay in range [-1, 1]
        # actions = torch.clamp(actions, -1.0, 1.0)
    
    return actions

def train_flow_matching(args):
    """
    Train Flow Matching model with configurable parameters
    
    Args:
        args: Command line arguments parsed by argparse
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters from command line arguments
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    save_dir = args.save_dir
    seq_length = args.seq_length
    pred_horizon = args.future_steps

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dir, f"logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Create dataset and data loader
    dataset = UAVNavDataset(data_dir, seq_length=seq_length, pred_horizon=pred_horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = FlowMatchingPolicy(history_length=seq_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for images, target_actions in dataloader:
            images = images.to(device)
            target_actions = target_actions.to(device)

            # Compute loss
            loss = flow_matching_loss(model, images, target_actions, device)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch+1)
        
        # Print training progress
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, "flow_matching_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save model
    final_model_path = os.path.join(save_dir, "flow_matching_policy_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    writer.close()
    
    print("Training complete!")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={log_dir}")

    save_loss_plot(log_dir, epochs, save_dir)
    
    return best_loss

def save_loss_plot(log_dir, epochs, save_dir):
    """Save loss curve plot"""
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    # Create plot directory
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Read TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get loss values
    if 'Loss/train_epoch' in event_acc.Tags()['scalars']:
        loss_events = event_acc.Scalars('Loss/train_epoch')
        epochs_list = [int(e.step) for e in loss_events]
        loss_values = [e.value for e in loss_events]
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, loss_values, 'b-', label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(plot_dir, 'training_loss_curve.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Loss curve saved to: {plot_path}")
    
    return plot_path

def evaluate_model(args):
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMatchingPolicy().to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "flow_matching_policy.pth")))
    model.eval()

    # Create test dataset
    test_dataset = UAVNavDataset(args.data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate model
    with torch.no_grad():
        for images, target_actions in test_dataloader:
            images = images.to(device)

            # Sample actions from the model
            predicted_actions = sample_action(model, images, num_steps=50, device=device)
            
            print(f"Target action: {target_actions.cpu().numpy()}")
            print(f"Predicted action: {predicted_actions.cpu().numpy()}")
            print("---")