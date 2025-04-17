import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.amp
from model import CarotidDataset, SimpleUNet, FusionViTNet
from utils import plot_training_results, save_checkpoint
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 4
epochs = 100
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------
# Dataset Setup (Fixed)
# ---------------------
transform = transforms.Compose([transforms.ToTensor()])
mask_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CarotidDataset(
    us_images_dir='Common Carotid Artery Ultrasound Images/US images/train',
    mask_images_dir='Common Carotid Artery Ultrasound Images/Expert mask images/train',
    transform=transform,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = CarotidDataset(
    us_images_dir='Common Carotid Artery Ultrasound Images/US images/val',
    mask_images_dir='Common Carotid Artery Ultrasound Images/Expert mask images/val',
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ---------------------
# Model Setup (Fixed)
# ---------------------
unet_model = SimpleUNet(in_channels=3).to(device)
fusionvit_model = FusionViTNet(in_channels=1, out_channels=1).to(device)

# ---------------------
# Optimizer and Loss Setup (Fixed)
# ---------------------
optimizer_unet = optim.Adam(unet_model.parameters(), lr=learning_rate)
optimizer_fusionvit = optim.Adam(fusionvit_model.parameters(), lr=learning_rate)

criterion = torch.nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler()  # Automatic Mixed Precision scaler

def create_run_folder(base_dir='checkpoints'):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('run')]
    
    if existing_runs:
        run_numbers = [int(d.replace('run', '')) for d in existing_runs if d.replace('run', '').isdigit()]
        next_run_number = max(run_numbers) + 1
    else:
        next_run_number = 1

    run_folder = os.path.join(base_dir, f'run{next_run_number}')
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

def train_model(model, optimizer, train_loader, val_loader, num_epochs=epochs, run_folder=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Initialize AMP GradScaler without specifying 'cuda'
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            masks = masks.float()
            if masks.ndim == 4 and masks.shape[1] == 3:
                masks = masks.mean(dim=1, keepdim=True)

            optimizer.zero_grad()

            # Mixed precision
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                masks_resized = F.interpolate(masks, size=outputs_resized.shape[2:], mode='nearest')
                loss = criterion(outputs_resized, masks_resized)

            # Backpropagation and optimizer step with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.float()
                if masks.ndim == 4 and masks.shape[1] == 3:
                    masks = masks.mean(dim=1, keepdim=True)

                with autocast():  # No need to specify 'cuda' explicitly
                    outputs = model(images)
                    outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                    masks_resized = F.interpolate(masks, size=outputs_resized.shape[2:], mode='nearest')
                    loss = criterion(outputs_resized, masks_resized)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            filename=os.path.join(run_folder, f'{model.__class__.__name__}_best.pth'))

        # Append the train loss for each epoch
        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Ensure the plot function is called after losses are populated
        plot_training_results(model, epoch + 1, train_losses, val_losses, run_folder)  # Pass the correct epoch count

    return train_losses, val_losses  # Return losses for evaluation later

# ---------------------
# Main Training Loop
# ---------------------
def main():
    # Ensure you define a function to create the output run folder
    run_folder = create_run_folder()

    # Training U-Net model
    print("Training U-Net model...")
    unet_train_losses, unet_val_losses = train_model(unet_model, optimizer_unet, train_loader, val_loader, run_folder=run_folder)

    # Training FusionViTNet model
    print("Training FusionViTNet model...")
    fusionvit_train_losses, fusionvit_val_losses = train_model(fusionvit_model, optimizer_fusionvit, train_loader, val_loader, run_folder=run_folder)

if __name__ == "__main__":
    main()
