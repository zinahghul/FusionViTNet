import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from torch.utils.data import DataLoader
from model import CarotidDataset, SimpleUNet, FusionViTNet
from utils import plot_confusion_matrix, plot_roc_curve, plot_metrics, calculate_all_metrics, save_confusion_matrix_and_roc
from torchvision import transforms

# Hyperparameters (same as training)
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------
# Dataset Setup
# ---------------------
transform = transforms.Compose([transforms.ToTensor()])

# Load the validation set for evaluation
val_dataset = CarotidDataset(
    us_images_dir='Common Carotid Artery Ultrasound Images/US images/val',
    mask_images_dir='Common Carotid Artery Ultrasound Images/Expert mask images/val',
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ---------------------
# Model Setup
# ---------------------
def load_model(model_class, checkpoint_path):
    model = model_class(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

unet_model = load_model(SimpleUNet, 'checkpoints/run12/unet_baseline_best.pth')
fusionvit_model = load_model(FusionViTNet, 'checkpoints/run12/fusionvitnet_best.pth')

# ---------------------
# Evaluation Functions
# ---------------------
def evaluate_model(model, val_loader, criterion, device, save_dir):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    metrics_dict = {}

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()

            outputs = model(images)
            outputs_resized = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            masks_resized = torch.nn.functional.interpolate(masks, size=outputs_resized.shape[2:], mode='nearest')

            val_loss += criterion(outputs_resized, masks_resized).item()
            all_preds.append(outputs_resized)
            all_targets.append(masks_resized)

    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    val_loss /= len(val_loader)

    metrics, metrics_values = calculate_all_metrics(model, val_loader, criterion, device)
    metrics['Val Loss'] = val_loss
    metrics_dict['metrics'] = metrics

    # Save confusion matrix and ROC curve
    save_confusion_matrix_and_roc(model, val_loader, device, save_dir, model_name="ModelName")

    return metrics_dict

def compute_metrics(all_preds, all_labels):
    # Flatten the arrays
    all_preds_flat = [item.flatten() for sublist in all_preds for item in sublist]
    all_labels_flat = [item.flatten() for sublist in all_labels for item in sublist]

    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels_flat, all_preds_flat)

    # Classification Report
    report = classification_report(all_labels_flat, all_preds_flat, target_names=["Background", "Carotid Artery"])

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels_flat, all_preds_flat)
    roc_auc = auc(fpr, tpr)

    return cm, report, fpr, tpr, roc_auc

def save_results(run_folder, cm, report, fpr, tpr, roc_auc):
    # Ensure the run_folder exists
    os.makedirs(run_folder, exist_ok=True)

    # Save confusion matrix
    plot_confusion_matrix(cm, os.path.join(run_folder, 'confusion_matrix.png'))
    
    # Save ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, os.path.join(run_folder, 'roc_curve.png'))

    # Save classification report
    with open(os.path.join(run_folder, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Optionally, save additional metrics or figures
    plot_metrics(run_folder)

def main():
    # Define your run folder where evaluation results will be saved
    run_folder = 'checkpoints/run1/evaluation_results'
    os.makedirs(run_folder, exist_ok=True)

    print("Evaluating U-Net Model...")
    unet_preds, unet_labels = evaluate_model(unet_model, val_loader)
    unet_cm, unet_report, unet_fpr, unet_tpr, unet_roc_auc = compute_metrics(unet_preds, unet_labels)
    save_results(run_folder, unet_cm, unet_report, unet_fpr, unet_tpr, unet_roc_auc)

    print("Evaluating FusionViTNet Model...")
    fusionvit_preds, fusionvit_labels = evaluate_model(fusionvit_model, val_loader)
    fusionvit_cm, fusionvit_report, fusionvit_fpr, fusionvit_tpr, fusionvit_roc_auc = compute_metrics(fusionvit_preds, fusionvit_labels)
    save_results(run_folder, fusionvit_cm, fusionvit_report, fusionvit_fpr, fusionvit_tpr, fusionvit_roc_auc)

if __name__ == "__main__":
    main()
