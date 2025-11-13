import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import math
from datetime import datetime
import time

class BCPolicy(nn.Module):
    """
    Behavior Cloning based UAV navigation policy network
    Input: image sequence [B, T, C, H, W]
    Output: action sequence [B, pred_horizon, action_dim]
    """
    def __init__(self, 
                 image_size=128, 
                 action_dim=2, 
                 hidden_dim=512, 
                 history_length=1, 
                 pred_horizon=1,
                 use_lstm=True):
        super(BCPolicy, self).__init__()
        self.action_dim = action_dim
        self.history_length = history_length
        self.pred_horizon = pred_horizon
        self.use_lstm = use_lstm
        
        # Visual encoder - using CNN to extract features from each frame
        self.visual_encoder = nn.Sequential(
            # Input: [B*T, C, H, W]
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Calculate visual feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            visual_features = self.visual_encoder(dummy_input)
            visual_feature_dim = visual_features.shape[1]
        
        # Temporal processing module
        if use_lstm:
            # Use LSTM for temporal sequence processing
            self.temporal_processor = nn.LSTM(
                input_size=visual_feature_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            temporal_output_dim = hidden_dim
        else:
            # Use simple fully connected layers for temporal processing
            self.temporal_processor = nn.Sequential(
                nn.Linear(visual_feature_dim * history_length, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            temporal_output_dim = hidden_dim
        
        # Action prediction head - outputs future action sequence
        self.action_head = nn.Sequential(
            nn.Linear(temporal_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * pred_horizon),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, image_sequence):
        """
        Args:
            image_sequence: input image sequence [B, history_length, C, H, W]
        
        Returns:
            predicted_actions: predicted action sequence [B, pred_horizon, action_dim]
        """
        batch_size, seq_len, C, H, W = image_sequence.shape
        
        # Encode each image
        visual_features = []
        for t in range(seq_len):
            img_t = image_sequence[:, t, :, :, :]  # [B, C, H, W]
            feat_t = self.visual_encoder(img_t)  # [B, visual_feature_dim]
            visual_features.append(feat_t)
        
        # Combine temporal features
        if self.use_lstm:
            # Process temporal sequence with LSTM [B, seq_len, visual_feature_dim]
            visual_features = torch.stack(visual_features, dim=1)
            temporal_features, _ = self.temporal_processor(visual_features)
            # Use features from the last time step
            temporal_output = temporal_features[:, -1, :]  # [B, hidden_dim]
        else:
            # Concatenate features from all time steps [B, seq_len * visual_feature_dim]
            visual_features = torch.cat(visual_features, dim=1)
            temporal_output = self.temporal_processor(visual_features)  # [B, hidden_dim]
        
        # Predict action sequence
        actions_flat = self.action_head(temporal_output)  # [B, action_dim * pred_horizon]
        predicted_actions = actions_flat.view(batch_size, self.pred_horizon, self.action_dim)  # [B, pred_horizon, action_dim]
        
        return predicted_actions

def bc_loss(predicted_actions, target_actions):
    """
    Calculate Behavior Cloning loss - Mean Squared Error
    """
    return F.mse_loss(predicted_actions, target_actions)

def train_bc_policy(args):
    """
    Train Behavior Cloning policy
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    save_dir = args.save_dir
    seq_length = args.seq_length
    pred_horizon = args.future_steps
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dir, f"bc_logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Create dataset and data loader - using the provided UAVNavDataset
    from core.datasets import UAVNavDataset
    dataset = UAVNavDataset(
        data_dir, 
        seq_length=seq_length, 
        pred_horizon=pred_horizon
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # Initialize model
    model = BCPolicy(
        image_size=args.image_size,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        history_length=seq_length,
        pred_horizon=pred_horizon,
        use_lstm=args.use_lstm
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for images, target_actions in dataloader:
            images = images.to(device)
            target_actions = target_actions.to(device)
            
            # Forward pass
            pred_actions = model(images)
            
            # Calculate loss
            loss = bc_loss(pred_actions, target_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch loss
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(dataloader) + num_batches)
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch+1)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch+1)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"bc_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, "bc_policy_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.6f}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "bc_policy_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    writer.close()
    
    print("Training complete!")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={log_dir}")
    
    # Save loss plot
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
        plt.title('BC Policy Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(plot_dir, 'bc_training_loss_curve.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Loss curve saved to: {plot_path}")
    
    return plot_path