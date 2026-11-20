import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from skimage.color import lab2rgb
import numpy as np
import warnings
import sys

# Bootstrap: make the shared modules (config, utils) importable from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils import load_dataset, visualize_lab_results
from colorization_model import ColorizationUNet

# Silence the scikit-image warning about Lab->sRGB conversions producing out-of-gamut values:
# this is expected with network-generated colors and is not critical.
warnings.filterwarnings(
    "ignore",
    message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in.*",
    category=UserWarning
)


def plot_and_save_loss(history, save_path):
    """Plot and save the combined (L1 + Perceptual) training and validation loss curves.

    :param history: dict with 'train_loss' and 'val_loss' lists
    :param save_path: path where the figure is saved
    :return: None
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Combined Loss (L1 + Perceptual)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"\n[INFO] Loss plot saved to: {save_path}")


class VGGPerceptualLoss(nn.Module):
    """Perceptual (content) loss comparing VGG19 feature maps instead of raw pixels."""

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Pretrained VGG19 up to relu5_4, used only as a frozen feature extractor.
        vgg = models.vgg19(pretrained=True).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg_slice = vgg
        self.criterion = nn.L1Loss()
        # ImageNet normalization stats, registered as buffers so they follow the module's device.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _lab_to_rgb(self, L, ab):
        """Convert a batch of Lab tensors to RGB (required before feeding VGG).

        :param L: L channel in [-1, 1], shape (N, 1, H, W)
        :param ab: a*b* channels in [-1, 1], shape (N, 2, H, W)
        :return: RGB tensor, shape (N, 3, H, W)
        """
        # Denormalize the channels back to their original Lab ranges.
        L = (L + 1.) * 50.
        ab = ab * 128.
        Lab = torch.cat([L, ab], dim=1)

        # Detach before converting to NumPy, which runs on CPU and cannot carry gradients.
        Lab_detached = Lab.detach().permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = [lab2rgb(img) for img in Lab_detached]
        return torch.from_numpy(np.stack(rgb_imgs, axis=0)).permute(0, 3, 1, 2).to(L.device).float()

    def forward(self, L_channel, pred_ab, true_ab):
        """Compute the perceptual loss between predicted and true colorizations.

        :param L_channel: shared L channel, shape (N, 1, H, W)
        :param pred_ab: predicted a*b* channels, shape (N, 2, H, W)
        :param true_ab: ground-truth a*b* channels, shape (N, 2, H, W)
        :return: L1 loss between the VGG feature maps of the two images
        """
        pred_rgb = self._lab_to_rgb(L_channel, pred_ab)
        true_rgb = self._lab_to_rgb(L_channel, true_ab)

        pred_rgb_norm = (pred_rgb - self.mean) / self.std
        true_rgb_norm = (true_rgb - self.mean) / self.std

        pred_features = self.vgg_slice(pred_rgb_norm)
        true_features = self.vgg_slice(true_rgb_norm)
        return self.criterion(pred_features, true_features)


def train_fn(model, loader, optimizer, l1_criterion, perceptual_criterion, l1_lambda, device):
    """Run a single training epoch with the combined L1 + Perceptual loss.

    :param model: the U-Net model
    :param loader: training DataLoader yielding (L, ab) batches
    :param optimizer: optimizer updating the model parameters
    :param l1_criterion: L1 loss on the a*b* channels
    :param perceptual_criterion: VGG perceptual loss
    :param l1_lambda: weight of the L1 term in the combined loss
    :param device: device the model runs on
    :return: average training loss over the epoch
    """
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, leave=True, desc="Training")
    for idx, (L_channel, ab_channels) in enumerate(loop):
        L_channel, ab_channels = L_channel.to(device), ab_channels.to(device)

        predicted_ab = model(L_channel)

        # Combined loss: weighted L1 on a*b* plus the perceptual term.
        loss_l1 = l1_criterion(predicted_ab, ab_channels)
        loss_perceptual = perceptual_criterion(L_channel, predicted_ab, ab_channels)
        loss = l1_lambda * loss_l1 + loss_perceptual
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(total_loss=loss.item(), l1=loss_l1.item(), perceptual=loss_perceptual.item())

    return running_loss / len(loader)


def main():
    """Orchestrate the full U-Net training process (data, loss, training loop, checkpoints).

    :return: None
    """
    config = Config()
    device = config.DEVICE

    model = ColorizationUNet(in_channels=1).to(device)
    train_loader, val_loader, test_loader = load_dataset(
        dataset_name=config.DATASET_NAME,
        batch_size=config.COLORIZER_BATCH_SIZE,
        subset_size=config.SUBSET_SIZE,
        data_dir=config.DATA_DIR
    )

    l1_criterion = nn.L1Loss()
    perceptual_criterion = VGGPerceptualLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.COLORIZER_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': []}

    # Resume from a checkpoint if one exists.
    if os.path.exists(config.COLORIZER_CHECKPOINT_PATH):
        print("[INFO] Loading checkpoint...")
        checkpoint = torch.load(config.COLORIZER_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        history = checkpoint['history']
        print(f"        Resuming from epoch {start_epoch + 1}")

    print("[INFO] Starting training with the combined loss (L1 + Perceptual)...")
    for epoch in range(start_epoch, config.COLORIZER_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.COLORIZER_EPOCHS} ---")
        train_loss = train_fn(model, train_loader, optimizer, l1_criterion, perceptual_criterion, config.L1_LAMBDA, device)

        # Validation loop.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            loop_val = tqdm(val_loader, leave=True, desc="Validating")
            for L_channel, ab_channels in loop_val:
                L_channel, ab_channels = L_channel.to(device), ab_channels.to(device)
                predicted_ab = model(L_channel)

                loss_l1_val = l1_criterion(predicted_ab, ab_channels)
                loss_perceptual_val = perceptual_criterion(L_channel, predicted_ab, ab_channels)
                loss = config.L1_LAMBDA * loss_l1_val + loss_perceptual_val
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"-> Epoch {epoch+1} results: Train Loss={train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)

        # Reduce the learning rate when the validation loss plateaus.
        scheduler.step(avg_val_loss)

        # Periodically save a sample visualization.
        if (epoch + 1) % config.COLORIZER_VIZ_EVERY_N_EPOCHS == 0:
            visualize_lab_results(model, test_loader, device, epoch, config.OUTPUT_DIR)

        # Save the best model based on the validation loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.COLORIZER_MODEL_PATH)
            print(f"  -> [Checkpoint] Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No validation improvement. Patience: {epochs_no_improve}/{config.COLORIZER_PATIENCE}")

        # Save a full checkpoint at the end of every epoch to allow resuming.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
        }, config.COLORIZER_CHECKPOINT_PATH)

        # Early stopping.
        if epochs_no_improve >= config.COLORIZER_PATIENCE:
            print(f"\n-> Early stopping: no improvement for {config.COLORIZER_PATIENCE} epochs.")
            break

    # Final loss curve.
    loss_plot_path = os.path.join(config.OUTPUT_DIR, "percep_loss_curve_1.png")
    plot_and_save_loss(history, loss_plot_path)


if __name__ == "__main__":
    main()
