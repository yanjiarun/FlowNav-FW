import os
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import time
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class UAVNavDataset(Dataset):
    """UAV navigation dataset with sequence support"""
    def __init__(self, data_dir, seq_length=1, pred_horizon=1, resize_shape=(256, 256), crop_shape=(224, 224), transform=None):
        """
        Args:
            data_dir (string): Directory with all the data
            seq_length (int): Number of consecutive images to use as input sequence
            pred_horizon (int): Number of future actions to predict
            resize_shape (tuple): Target size for image resizing (width, height)
            crop_shape (tuple): Target size for image cropping (width, height)
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.transform = transform

        self.preprocess = transforms.Compose([
            transforms.Resize(resize_shape) if resize_shape else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(crop_shape) if crop_shape else transforms.Lambda(lambda x: x),
        ])

        # Load action data
        self.actions = np.load(os.path.join(data_dir, "commands.npy"))  # [N,]
        
        # Convert angles to 2D vectors [cos(θ), sin(θ)]
        self.actions_vector = np.zeros((len(self.actions), 2))
        for i, angle_deg in enumerate(self.actions):
            angle_rad = math.radians(angle_deg)
            self.actions_vector[i, 0] = math.cos(angle_rad)
            self.actions_vector[i, 1] = math.sin(angle_rad)
        
        # Get list of image files
        self.img_dir = os.path.join(data_dir, "imgs")
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir) 
            if f.startswith("image_") and f.endswith(".png")
        ])
        
        # Preload all images into memory
        self.images_tensor = []
        print("Starting image preloading...")
        for img_file in self.img_files:
            img_path = os.path.join(self.img_dir, img_file)
            # Convert to RGB and store as PIL Image
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
            
            # Apply transformation
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default transformation: PIL image -> Tensor
                image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0
            self.images_tensor.append(image_tensor)
        print(f"Successfully preloaded {len(self.images_tensor)} images")

        # Precompute all possible sequence indices
        self.sequence_indices = []
        # Ensure we have enough future actions for prediction
        for i in range(len(self.images_tensor) - self.seq_length - self.pred_horizon + 1):
            self.sequence_indices.append(i)
        
        # Ensure data consistency
        assert len(self.images_tensor) == len(self.actions), \
            f"Number of images ({len(self.images_tensor)}) doesn't match number of actions ({len(self.actions)})"
        
        print(f"Action angle range: [{self.actions.min():.1f}°, {self.actions.max():.1f}°]")
        print(f"Action vector range: [{self.actions_vector.min():.3f}, {self.actions_vector.max():.3f}]")
        print(f"Dataset size: {len(self)} sequences")
    
    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        # Get the actual start index from precomputed list
        start_idx = self.sequence_indices[idx]
        
        # Extract image sequence
        image_sequence = []
        for i in range(start_idx, start_idx + self.seq_length):
            image_sequence.append(self.images_tensor[i])
        image_sequence = torch.stack(image_sequence)  # [seq_length, C, H, W]
        
        # Extract action sequence (future actions)
        action_sequence = []
        # Start from the end of the image sequence
        action_start_idx = start_idx + self.seq_length
        for i in range(action_start_idx, action_start_idx + self.pred_horizon):
            action_sequence.append(self.actions_vector[i])
        action_sequence = torch.FloatTensor(np.array(action_sequence))  # [pred_horizon, 2]
        
        return image_sequence, action_sequence

# Optional data augmentation transforms
def get_default_transform(image_size=128):
    """
    Get default data preprocessing transforms
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    # Test dataset
    data_dir = "../data/data_flow_nav/episode_1"

    dataset = UAVNavDataset(data_dir, seq_length=1)
    print(f"Dataset size: {len(dataset)}")
    image, action = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action value: {action}")
    print(f"Image min/max: {image.min():.3f}, {image.max():.3f}")

    import matplotlib.pyplot as plt
    plt.imshow(image.numpy().transpose(1, 2, 0))
    plt.title("Current Image")
    plt.axis('off')  # Turn off axis display
    plt.show()
    print("Visualization complete.")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)