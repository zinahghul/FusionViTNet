import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF
import cv2

# ---------------------
# Save Checkpoint Function
# ---------------------
def save_checkpoint(model, optimizer, epoch, val_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filename)

def plot_confusion_matrix(cm, filepath):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, filepath):
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(filepath)
    plt.close()

def plot_metrics(run_folder):
    # Example of plotting additional metrics
    metrics = {'Dice': 0.9, 'IoU': 0.8, 'Precision': 0.85}  # Example metrics
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Evaluation Metrics')
    plt.savefig(os.path.join(run_folder, 'evaluation_metrics.png'))
    plt.close()

# ---------------------
# Plot Training Results
# --------------------

def plot_training_results(model, epoch, train_losses, val_losses, run_folder):
    if len(train_losses) == 0 or len(val_losses) == 0:
        print("No data to plot.")
        return

    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model.__class__.__name__}')
    plt.legend()
    plt.savefig(os.path.join(run_folder, 'training_loss.png'))
    plt.close()

# ---------------------
# Plot Metrics Comparison
# ---------------------
def plot_metrics_comparison(unet_metrics, fusionvit_metrics, run_folder):
    metrics = ["dice", "iou", "precision", "sensitivity", "specificity", "mse"]
    for metric in metrics:
        plt.figure()
        unet_values = [m[metric] for m in unet_metrics if metric in m]
        fusionvit_values = [m[metric] for m in fusionvit_metrics if metric in m]
        if len(unet_values) > 0:
            plt.plot(range(1, len(unet_values) + 1), unet_values, label='U-Net')
        if len(fusionvit_values) > 0:
            plt.plot(range(1, len(fusionvit_values) + 1), fusionvit_values, label='FusionViTNet')
        plt.title(f"{metric.upper()} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_folder, f"{metric}_comparison.png"))
        plt.close()

def plot_loss_curve(train_losses, val_losses, model_name, run_folder):
    if len(train_losses) == len(val_losses) and len(train_losses) > 0:
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.title(f"{model_name} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        filename = os.path.join(run_folder, f"loss_{model_name.lower()}.png")
        plt.savefig(filename)
        plt.close()

def plot_combined_loss_curve(unet_train_losses, unet_val_losses, fusionvit_train_losses, fusionvit_val_losses, run_folder):
    min_len = min(len(unet_train_losses), len(unet_val_losses), len(fusionvit_train_losses), len(fusionvit_val_losses))
    if min_len > 0:
        plt.figure()
        plt.plot(range(1, min_len + 1), unet_train_losses[:min_len], label='U-Net Train Loss', linestyle='--')
        plt.plot(range(1, min_len + 1), unet_val_losses[:min_len], label='U-Net Val Loss')
        plt.plot(range(1, min_len + 1), fusionvit_train_losses[:min_len], label='FusionViTNet Train Loss', linestyle='--')
        plt.plot(range(1, min_len + 1), fusionvit_val_losses[:min_len], label='FusionViTNet Val Loss')
        plt.title("Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_folder, "loss_comparison.png"))
        plt.close()

# ---------------------
# Overlay Predictions Function
# ---------------------
def overlay_predictions(model, dataloader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            if idx >= 5:  # Limit to first 5 images
                break
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy().round()
            masks = masks.cpu().numpy().round()

            for i in range(images.size(0)):
                overlay = overlay_mask_on_image(images[i], torch.tensor(preds[i]))
                plt.figure(figsize=(8, 8))
                plt.imshow(overlay)
                plt.title(f'Overlay - {idx}_{i}')
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, f'overlay_{idx}_{i}.png'))
                plt.close()


# ---------------------
# Overlay Mask on Image (Utility for overlaying predictions on input image)
# ---------------------
def overlay_mask_on_image(model, val_loader, device, save_dir, max_images=10):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        img_count = 0  # Track total images saved
        for batch_idx, (images, masks) in enumerate(val_loader):
            if img_count >= max_images:
                break

            images = images.to(device)
            masks = masks.to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

            for i in range(images.size(0)):
                if img_count >= max_images:
                    break

                img = images[i].cpu()
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()

                # Convert grayscale image to RGB
                img_pil = TF.to_pil_image(img)
                overlay = np.array(img_pil.convert("RGB"))

                # Resize mask and pred to match overlay if needed
                if mask.shape != overlay.shape[:2]:
                    mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                if pred.shape != overlay.shape[:2]:
                    pred = cv2.resize(pred, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Overlay: green = ground truth, red = prediction
                overlay[mask > 0.5] = [0, 255, 0]
                overlay[pred > 0.5] = [255, 0, 0]

                # Plot and save
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
                axs[0].set_title("Input Image")
                axs[1].imshow(mask, cmap='gray')
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred, cmap='gray')
                axs[2].set_title("Prediction")
                axs[3].imshow(overlay)
                axs[3].set_title("Overlay")

                for ax in axs:
                    ax.axis('off')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f"sample_{batch_idx}_{i}.png")
                plt.savefig(save_path)
                plt.close()

                img_count += 1

def plot_roc_curve(y_true, y_pred, save_path, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_metrics_table(metrics_dict, save_path):
    df = pd.DataFrame(metrics_dict).T.round(4)
    df = df.reset_index().rename(columns={'index': 'Model'})

    # Save as CSV only
    csv_path = os.path.join(save_path, 'final_metrics.csv')
    df.to_csv(csv_path, index=False)


def calculate_all_metrics(model, val_loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            # Resize predictions to match target shape
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    preds = np.round(all_preds)

    epsilon = 1e-7
    tp = np.sum((preds == 1) & (all_targets == 1))
    tn = np.sum((preds == 0) & (all_targets == 0))
    fp = np.sum((preds == 1) & (all_targets == 0))
    fn = np.sum((preds == 0) & (all_targets == 1))

    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    mse = np.mean((preds - all_targets) ** 2)
    mAP = average_precision_score(all_targets.flatten(), all_preds.flatten())

    # Full dictionary for logging/saving
    metrics_dict = {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'MSE': mse,
        'mAP': mAP
    }

    # Scalar values for plotting
    metrics_values = [iou, dice, precision, sensitivity, specificity, mse, mAP]

    return metrics_dict, metrics_values

# Function to save confusion matrix and ROC curve
def save_confusion_matrix_and_roc(model, val_loader, device, save_dir, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs).view(-1).cpu().numpy()
            labels = masks.view(-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Binarize
    binary_preds = np.array(all_preds) > 0.5
    binary_labels = np.array(all_labels)

    # Confusion Matrix
    cm = confusion_matrix(binary_labels, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(binary_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"))
    plt.close()

    return fpr, tpr, roc_auc


# Function to save predictions and masks
def save_predictions_and_masks(model, val_loader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            masks_resized = F.interpolate(masks, size=outputs_resized.shape[2:], mode='nearest')
            
            preds = torch.sigmoid(outputs_resized).cpu().numpy()
            masks_np = masks_resized.cpu().numpy()

            batch_size = preds.shape[0]
            for i in range(batch_size):
                pred_img = np.squeeze(preds[i])  # Shape: (H, W)
                mask_img = np.squeeze(masks_np[i])  # Shape: (H, W)

                pred_path = os.path.join(save_dir, f"pred_{batch_idx}_{i}.png")
                mask_path = os.path.join(save_dir, f"mask_{batch_idx}_{i}.png")

                plt.imsave(pred_path, pred_img, cmap='gray')
                plt.imsave(mask_path, mask_img, cmap='gray')


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            if masks.ndim == 4 and masks.shape[1] == 3:
                masks = masks.mean(dim=1, keepdim=True)

            outputs = model(images)
            outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            masks_resized = F.interpolate(masks, size=outputs_resized.shape[2:], mode='nearest')

            val_loss += criterion(outputs_resized, masks_resized).item()
            all_preds.append(outputs_resized)
            all_targets.append(masks_resized)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    val_loss /= len(dataloader)
    metrics = calculate_all_metrics(all_preds, all_targets)
    metrics['Val Loss'] = val_loss
    return metrics


def plot_comparison(metrics_dict, save_dir):
    metrics_names = list(next(iter(metrics_dict.values())).keys())
    for metric in metrics_names:
        values = [metrics_dict[model][metric] for model in metrics_dict]
        plt.figure()
        plt.bar(metrics_dict.keys(), values)
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric.lower().replace(" ", "_")}_comparison.png'))
        plt.close()

    # Save metrics table as PNG and CSV
    save_metrics_table(metrics_dict, save_dir)
