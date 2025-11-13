import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import time
from core.datasets import UAVNavDataset
from core.resnet import get_resnet, replace_bn_with_gn

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import logging
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

logger = logging.getLogger(__name__)

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding"""
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

class Conv1dBlock(nn.Module):
    """1D convolutional block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.block(x)

class Downsample1d(nn.Module):
    """1D downsampling"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    """1D upsampling"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class ConditionalResidualBlock1D(nn.Module):
    """Conditional 1D residual block"""
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation (https://arxiv.org/abs/1709.07871)
        # Predicts scale and bias for each channel
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('b t -> b t 1'),
        )

        # Ensure dimension compatibility
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x : [batch_size, in_channels, horizon]
            cond : [batch_size, cond_dim]
        
        Returns:
            out : [batch_size, out_channels, horizon]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    """1D Conditional UNet diffusion model"""
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        self.input_dim = input_dim
        self.local_cond_dim = local_cond_dim
        self.global_cond_dim = global_cond_dim
        
        # All dimensions
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Diffusion timestep encoder
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.SiLU(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # Condition dimension
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local condition encoder
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            self.local_cond_encoder = nn.ModuleList([
                # Downsample encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # Upsample encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        # Middle modules
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # Downsample modules
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        self.down_modules = down_modules

        # Upsample modules
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        self.up_modules = up_modules
        
        # Final convolution layer
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        logger.info(
            "Number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: torch.Tensor, 
            local_cond=None, 
            global_cond=None):
        """
        Args:
            sample: input sample [B, T, input_dim]
            timestep: diffusion timestep [B]
            local_cond: local condition [B, T, local_cond_dim]
            global_cond: global condition [B, global_cond_dim]
        
        Returns:
            output: [B, T, input_dim]
        """
        # Rearrange dimensions: [B, T, input_dim] -> [B, input_dim, T]
        sample = einops.rearrange(sample, 'b t d -> b d t')
        
        # Process timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        
        # Diffusion timestep embedding
        global_feature = self.diffusion_step_encoder(timestep)
        
        # Add global condition
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)
        
        # Encode local features
        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            # Rearrange: [B, T, local_cond_dim] -> [B, local_cond_dim, T]
            local_cond = einops.rearrange(local_cond, 'b t d -> b d t')
            resnet, resnet2 = self.local_cond_encoder
            x_local = resnet(local_cond, global_feature)
            h_local.append(x_local)
            x_local = resnet2(local_cond, global_feature)
            h_local.append(x_local)
        
        # Downsample path
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            # Add local condition
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        
        # Middle blocks
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        
        # Upsample path
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # Add local condition
            if idx == len(self.up_modules) - 1 and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Rearrange dimensions: [B, input_dim, T] -> [B, T, input_dim]
        x = einops.rearrange(x, 'b d t -> b t d')
        return x

class DiffusionPolicy(nn.Module):
    """Diffusion-based UAV navigation policy"""
    def __init__(self, 
                 history_length=1, 
                 pred_horizon=16,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 num_diffusion_timesteps=1000):
        super().__init__()
        self.action_dim = 2  # Heading vector dimension
        self.history_length = history_length
        self.pred_horizon = pred_horizon
        self.num_diffusion_timesteps = num_diffusion_timesteps
        
        # Visual encoder
        self.visual_encoder = get_resnet('resnet18')
        self.visual_encoder = replace_bn_with_gn(self.visual_encoder)
        vision_feature_dim = 512 * history_length
        
        # Diffusion model - Key modification: input_dim changed to action dimension
        self.diffusion_model = ConditionalUnet1D(
            input_dim=self.action_dim,  # Changed to action dimension, not sequence length
            global_cond_dim=vision_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims
        )
        
        # Diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
                                    num_train_timesteps=100, # Number of diffusion training steps
                                    beta_start=0.0001, # Initial beta value
                                    beta_end=0.02, # Final beta value
                                    beta_schedule="squaredcos_cap_v2", # Beta scheduling strategy
                                    variance_type="fixed_small", # Variance type
                                    clip_sample=True, # Whether to clip samples
                                    prediction_type="epsilon" # Prediction type
                                    )

    def predict_action(self, images, num_inference_steps=100):
        """Sample action sequence from diffusion model"""
        B = images.shape[0]
        device = images.device
        
        # Extract visual features
        sequence_features = self.visual_encoder(images.flatten(end_dim=1))
        sequence_features = sequence_features.reshape(B, self.history_length, -1)
        global_cond = sequence_features.flatten(start_dim=1)  # [B, history_length*512]
        
        # Initialize random noise - maintain 3D structure [B, T, Da]
        noisy_actions = torch.randn(
            (B, self.pred_horizon, self.action_dim), 
            device=device
        )
        
        # Set inference timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # Iterative denoising
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise - input maintains 3D structure
            noise_pred = self.diffusion_model(
                noisy_actions, 
                timesteps, 
                global_cond=global_cond
            )
            
            # Denoising step
            noisy_actions = self.noise_scheduler.step(noise_pred, t, noisy_actions).prev_sample
        
        return noisy_actions  # Already [B, T, Da]

def diffusion_loss(model, images, target_actions, device):
    """Diffusion model loss function (corrected version)"""
    B, T, Da = target_actions.shape
    assert T == model.pred_horizon
    
    # No need to flatten action sequence, maintain 3D [B, T, Da]
    
    # Extract visual features
    sequence_features = model.visual_encoder(images.flatten(end_dim=1))
    sequence_features = sequence_features.reshape(B, model.history_length, -1)
    global_cond = sequence_features.flatten(start_dim=1)  # [B, history_length*512]
    
    # Sample random timesteps
    timesteps = torch.randint(
        0, model.noise_scheduler.config.num_train_timesteps, 
        (B,), device=device
    ).long()
    
    # Sample noise - maintain 3D
    noise = torch.randn_like(target_actions)
    
    # Add noise to actions
    noisy_actions = model.noise_scheduler.add_noise(
        target_actions, noise, timesteps)
    
    # Predict noise - input maintains 3D
    noise_pred = model.diffusion_model(
        noisy_actions, 
        timesteps, 
        global_cond=global_cond
    )
    
    # Calculate loss
    loss = F.mse_loss(noise_pred, noise)
    return loss

def train_diffusion_policy(args):
    """Train diffusion policy model"""
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
    log_dir = os.path.join(save_dir, f"diffusion_logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Create dataset
    dataset = UAVNavDataset(
        data_dir, 
        seq_length=seq_length, 
        pred_horizon=pred_horizon
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = DiffusionPolicy(
        history_length=seq_length,
        pred_horizon=pred_horizon
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for images, target_actions in dataloader:
            images = images.to(device)
            target_actions = target_actions.to(device)  # [B, T, Da]
            
            # Calculate loss
            loss = diffusion_loss(model, images, target_actions, device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        writer.add_scalar('Loss/train', avg_loss, epoch+1)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"diffusion_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, "diffusion_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save final model
    final_model_path = os.path.join(save_dir, "diffusion_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    writer.close()
    print("Training complete!")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={log_dir}")
    
    # Save loss plot
    save_loss_plot(log_dir, epochs, save_dir)
    
    return best_loss

def save_loss_plot(log_dir, epochs, save_dir):
    """Save training loss curve plot"""
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    # Create plot directory
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Read TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get loss values
    if 'Loss/train' in event_acc.Tags()['scalars']:
        loss_events = event_acc.Scalars('Loss/train')
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

def evaluate_diffusion_model(args):
    """Evaluate diffusion model performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DiffusionPolicy(
        history_length=args.seq_length,
        pred_horizon=args.future_steps
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "diffusion_best.pth")))
    model.eval()
    
    # Create test dataset
    test_dataset = UAVNavDataset(
        args.data_dir, 
        seq_length=args.seq_length, 
        pred_horizon=args.future_steps,
        train=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluation metrics
    position_errors = []
    
    with torch.no_grad():
        for images, target_sequence in test_dataloader:
            images = images.to(device)
            target_sequence = target_sequence.to(device)  # [1, T, Da]
            
            # Sample action sequence
            predicted_sequence = model.predict_action(images)  # [1, T, Da]
            
            # Calculate sequence error (MSE)
            sequence_error = F.mse_loss(predicted_sequence, target_sequence).item()
            position_errors.append(sequence_error)
    
    # Print evaluation results
    avg_position_error = np.mean(position_errors)
    print(f"Average Sequence MSE: {avg_position_error:.4f}")
    
    return {'sequence_mse': avg_position_error}

def deploy_diffusion_policy(model_path, camera, controller, seq_length=5, pred_horizon=16):
    """Deploy diffusion policy to UAV"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DiffusionPolicy(
        history_length=seq_length,
        pred_horizon=pred_horizon
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize history buffer
    from collections import deque
    history_buffer = deque(maxlen=seq_length)
    action_buffer = deque(maxlen=pred_horizon)
    current_action_index = 0
    
    while True:
        # Get current image
        current_image = camera.capture()
        history_buffer.append(current_image)
        
        if len(history_buffer) == seq_length:
            if current_action_index >= len(action_buffer):
                # Need to generate new action sequence
                image_seq = np.stack(history_buffer, axis=0)
                image_tensor = torch.from_numpy(image_seq).float().unsqueeze(0).to(device)
                
                # Sample action sequence
                with torch.no_grad():
                    action_sequence = model.predict_action(image_tensor)
                    action_sequence = action_sequence.squeeze(0).cpu().numpy()
                
                # Reset action buffer
                action_buffer = deque(action_sequence)
                current_action_index = 0
            
            # Get current action
            current_action = action_buffer[current_action_index]
            current_action_index += 1
            
            # Convert to heading angle
            heading_angle = vector_to_heading(current_action)
            
            # Send control command
            controller.set_heading(heading_angle)
        
        time.sleep(0.1)  # Control frequency

def vector_to_heading(vector):
    """Convert 2D vector to heading angle (0-360 degrees)"""
    x, y = vector
    heading_rad = math.atan2(y, x)
    heading_deg = math.degrees(heading_rad) % 360
    return heading_deg

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='Train Diffusion Policy for UAV Navigation')
#     parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
#     parser.add_argument('--save_dir', type=str, default='./diffusion_models', help='Directory to save models')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
#     parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
#     parser.add_argument('--seq_length', type=int, default=5, help='Input sequence length (history frames)')
#     parser.add_argument('--future_steps', type=int, default=16, help='Prediction horizon (future steps)')
#     parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
#     parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
#     args = parser.parse_args()
    
#     train_diffusion_policy(args)