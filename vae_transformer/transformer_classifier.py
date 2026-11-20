import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import sys
from skimage.color import lab2rgb
import matplotlib.pyplot as plt

# Bootstrap: make the shared modules (config, utils) importable from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from autoencoder_module import VAE_NoSkips, load_dataset


# --- 1. Lab color-space quantization ---

class ColorQuantizer:
    """Discretizes the a*b* color space, turning colorization into a classification over a color palette."""

    def __init__(self, bins_per_axis=20, ab_range=(-100, 100)):
        """
        :param bins_per_axis: number of bins per axis (a* and b*); total classes = bins_per_axis^2
        :param ab_range: value range of the a* and b* axes
        """
        self.bins_per_axis = bins_per_axis
        self.num_classes = bins_per_axis * bins_per_axis

        # Palette = cartesian product of the per-axis bin centers: a [num_classes, 2] lookup table.
        bin_centers = torch.linspace(ab_range[0], ab_range[1], bins_per_axis)
        self.palette = torch.cartesian_prod(bin_centers, bin_centers)

    def to(self, device):
        """Move the color palette to the given device.

        :param device: target device (e.g. 'cuda')
        :return: self
        """
        self.palette = self.palette.to(device)
        return self

    def quantize_ab(self, ab_values):
        """Map continuous a*b* pixels to their nearest palette index.

        :param ab_values: a*b* tensor of shape [B, 2, H, W]
        :return: tensor of class indices of shape [B, H, W]
        """
        B, _, H, W = ab_values.shape
        pixels_ab = ab_values.permute(0, 2, 3, 1).reshape(-1, 2)

        # Squared Euclidean distance between every pixel and every palette color (via broadcasting).
        distances = torch.sum((pixels_ab.unsqueeze(1) - self.palette.unsqueeze(0)) ** 2, dim=2)
        min_indices = torch.argmin(distances, dim=1)
        return min_indices.reshape(B, H, W)

    def dequantize_ab(self, indices):
        """Convert class indices back to their a*b* palette colors.

        :param indices: tensor of class indices, shape [B]
        :return: a*b* values of shape [B, 2]
        """
        return self.palette[indices]


# --- 2. Transformer model ---

class ColorizationTransformer(nn.Module):
    """Transformer encoder predicting a probability distribution over the quantized color palette from a latent vector z."""

    def __init__(self, latent_dim, num_classes, nhead=8, num_layers=6, ff_dim=1024):
        """
        :param latent_dim: dimensionality of the input latent vector
        :param num_classes: number of color classes to predict
        :param nhead: number of attention heads
        :param num_layers: number of Transformer encoder layers
        :param ff_dim: internal/feed-forward dimension of the Transformer
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Project the latent vector z to the Transformer's internal dimension (d_model).
        self.input_proj = nn.Linear(latent_dim, ff_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head producing the logits for each color class.
        self.output_head = nn.Linear(ff_dim, num_classes)

    def forward(self, z):
        """
        :param z: latent vector, shape (N, latent_dim)
        :return: per-class logits, shape (N, num_classes)
        """
        x = self.input_proj(z)
        x = x.unsqueeze(1)                         # add a sequence dimension of length 1
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.squeeze(1)
        return self.output_head(transformer_out)


# --- 3. Training logic ---

def train_transformer(config, vae_model, transformer_model, quantizer):
    """Train the Transformer using a frozen, pretrained VAE as feature extractor.

    :param config: Config object with the training settings
    :param vae_model: pretrained VAE whose encoder provides the latent vector
    :param transformer_model: the Transformer to train
    :param quantizer: ColorQuantizer defining the target color classes
    :return: the trained Transformer model
    """
    device = config.DEVICE

    # Freeze the VAE: it is only used to encode the images, not trained.
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False

    transformer_model.to(device)
    vae_model.to(device)
    quantizer.to(device)

    train_loader, val_loader, test_loader = load_dataset(
        dataset_name=config.DATASET_NAME,
        batch_size=config.FULL_MODULE_BATCH_SIZE,
        subset_size=config.SUBSET_SIZE,
        data_dir=config.DATA_DIR
    )

    optimizer = optim.Adam(transformer_model.parameters(), lr=config.TRANSFORMER_LEARNING_RATE)
    # KL divergence is the natural loss for comparing two probability distributions.
    criterion = nn.KLDivLoss(reduction='batchmean')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Checkpointing / resume logic.
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "transformer_checkpoint.pth")
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Checkpoint found at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        transformer_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"        Resuming from epoch {start_epoch + 1}.")
    else:
        print("[INFO] No checkpoint found. Starting training from scratch.")

    print("\n[INFO] Starting Transformer training...")

    for epoch in range(start_epoch, config.TRANSFORMER_EPOCHS):
        transformer_model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.TRANSFORMER_EPOCHS} [Train]")
        for grayscale_img, ab_channels in loop:
            grayscale_img, ab_channels = grayscale_img.to(device), ab_channels.to(device)

            # INPUT path: generate the latent vector z (only mu is used as the semantic embedding).
            x_vae = (grayscale_img * 0.5) + 0.5
            mu, _, _ = vae_model.encoder(x_vae)

            # TARGET path: build the ground-truth color distribution.
            with torch.no_grad():
                # The dataloader provides a*b* normalized in [-1, 1] (divided by 128):
                # scale back to the original Lab range before quantizing.
                ab_gt = ab_channels * 128.0
                target_bins = quantizer.quantize_ab(ab_gt)
                # Target distribution = normalized histogram of color classes per image.
                target_dist = torch.zeros(grayscale_img.size(0), quantizer.num_classes, device=device)
                for i in range(grayscale_img.size(0)):
                    hist = torch.bincount(target_bins[i].flatten(), minlength=quantizer.num_classes)
                    target_dist[i] = hist.float() / (hist.sum() + 1e-8)

            optimizer.zero_grad()
            predicted_logits = transformer_model(mu)
            # KLDivLoss expects log-probabilities as input.
            predicted_log_probs = F.log_softmax(predicted_logits, dim=1)
            loss = criterion(predicted_log_probs, target_dist)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * grayscale_img.size(0)
            loop.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation loop (same logic, no backprop).
        transformer_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.TRANSFORMER_EPOCHS} [Validation]")
            for grayscale_img, ab_channels in loop_val:
                grayscale_img, ab_channels = grayscale_img.to(device), ab_channels.to(device)
                x_vae = (grayscale_img * 0.5) + 0.5
                mu, _, _ = vae_model.encoder(x_vae)

                ab_gt = ab_channels * 128.0
                target_bins = quantizer.quantize_ab(ab_gt)
                target_dist = torch.zeros(grayscale_img.size(0), quantizer.num_classes, device=device)
                for i in range(grayscale_img.size(0)):
                    hist = torch.bincount(target_bins[i].flatten(), minlength=quantizer.num_classes)
                    target_dist[i] = hist.float() / (hist.sum() + 1e-8)

                predicted_logits = transformer_model(mu)
                predicted_log_probs = F.log_softmax(predicted_logits, dim=1)
                loss = criterion(predicted_log_probs, target_dist)
                val_loss += loss.item() * grayscale_img.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"\n-> Epoch {epoch+1}: Train Loss={epoch_loss:.6f} | Val Loss={epoch_val_loss:.6f}")
        scheduler.step(epoch_val_loss)

        # Save the best model.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(transformer_model.state_dict(), config.TRANSFORMER_MODEL_PATH)
            print(f"  -> [Best Model] Best Transformer saved (Val Loss: {best_val_loss:.6f})")

        # Save a full checkpoint for resuming.
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        print(f"  -> [Checkpoint] Epoch {epoch+1} saved.")

        # Periodic visualization.
        if (epoch + 1) % config.VIZ_EVERY_N_EPOCHS == 0 or (epoch + 1) == config.TRANSFORMER_EPOCHS:
            print(f"\n[INFO] Generating visualization for epoch {epoch+1}...")
            visualize_colorization_results(
                config=config, vae_model=vae_model, transformer_model=transformer_model,
                quantizer=quantizer, data_loader=test_loader, epoch=epoch
            )

    return transformer_model


# --- 4. Visualization ---

def visualize_colorization_results(config, vae_model, transformer_model, quantizer, data_loader, num_images=5, epoch=None):
    """Colorize a batch using the Transformer's most likely class (argmax) and save the comparison grid.

    :param config: Config object
    :param vae_model: pretrained VAE providing the latent vector
    :param transformer_model: trained Transformer
    :param quantizer: ColorQuantizer mapping class indices to a*b* colors
    :param data_loader: loader providing (grayscale, ab) batches
    :param num_images: number of images to display
    :param epoch: current epoch index, used in the title/filename (None for final results)
    :return: None
    """
    device = config.DEVICE
    vae_model.eval()
    transformer_model.eval()

    grayscale_imgs, color_imgs_ab = next(iter(data_loader))
    grayscale_imgs = grayscale_imgs.to(device)[:num_images]
    color_imgs_ab = color_imgs_ab[:num_images]

    with torch.no_grad():
        x_vae = (grayscale_imgs * 0.5) + 0.5
        mu, _, _ = vae_model.encoder(x_vae)
        logits = transformer_model(mu)
        # Winner-takes-all: pick the most likely color class.
        predicted_indices = torch.argmax(logits, dim=1)
        predicted_ab = quantizer.dequantize_ab(predicted_indices)

    L_channel = (x_vae * 100).cpu().numpy().squeeze(1)

    # Colorized output: apply the single predicted tint to the whole image.
    colorized_images_rgb = []
    for i in range(num_images):
        lab_img = np.zeros((config.RESIZE_DIM, config.RESIZE_DIM, 3))
        lab_img[:, :, 0] = L_channel[i]
        lab_img[:, :, 1] = predicted_ab[i, 0].cpu().item()
        lab_img[:, :, 2] = predicted_ab[i, 1].cpu().item()
        colorized_images_rgb.append(np.clip(lab2rgb(lab_img), 0, 1))

    original_grayscale = x_vae.cpu().squeeze(1).numpy()
    # Ground truth: combine the L channel with the real a*b* channels from the dataloader.
    ground_truth_rgb = []
    for i in range(num_images):
        lab_gt = np.zeros((config.RESIZE_DIM, config.RESIZE_DIM, 3))
        lab_gt[:, :, 0] = L_channel[i]
        lab_gt[:, :, 1:] = (color_imgs_ab[i] * 128.0).permute(1, 2, 0).cpu().numpy()
        ground_truth_rgb.append(np.clip(lab2rgb(lab_gt), 0, 1))

    # 3 rows: input, colorized output, ground truth.
    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    fig.suptitle(f"Epoch: {epoch+1}" if epoch is not None else "Final Results", fontsize=16)

    for i in range(num_images):
        axs[0, i].imshow(original_grayscale[i], cmap='gray'); axs[0, i].set_title("Grayscale Input"); axs[0, i].axis('off')
        axs[1, i].imshow(colorized_images_rgb[i]); axs[1, i].set_title("Colorized Output"); axs[1, i].axis('off')
        axs[2, i].imshow(ground_truth_rgb[i]); axs[2, i].set_title("Ground Truth"); axs[2, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"transformer_colorization_epoch_{epoch+1:03d}.png" if epoch is not None else "transformer_colorization_final_results.png"
    save_path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"        Colorization results saved to: {save_path}")


# --- 5. Entry point ---

if __name__ == "__main__":
    # Load the pretrained VAE, initialize the quantizer and Transformer, train, then visualize.
    config = Config()
    device = config.DEVICE

    print("[INFO] Loading the pretrained VAE...")
    vae_model = VAE_NoSkips(latent_dim=config.AE_LATENT_DIM).to(device)
    try:
        vae_model.load_state_dict(torch.load(config.AE_MODEL_PATH, map_location=device))
        print(f"[INFO] VAE model loaded from: {config.AE_MODEL_PATH}")
    except FileNotFoundError:
        print(f"[ERROR] VAE model file not found at '{config.AE_MODEL_PATH}'. Aborting.")
        exit()

    color_quantizer = ColorQuantizer(bins_per_axis=20).to(device)
    transformer = ColorizationTransformer(
        latent_dim=config.AE_LATENT_DIM,
        num_classes=color_quantizer.num_classes,
        nhead=config.TRANSFORMER_NHEAD,
        num_layers=config.TRANSFORMER_NUM_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM
    ).to(device)

    print("\n[INFO] Transformer architecture:")
    print(transformer)

    trained_transformer = train_transformer(config, vae_model, transformer, color_quantizer)
    print("\n[INFO] Training completed.")

    # Final visualization with the best saved model.
    print("[INFO] Loading the best model for the final visualization...")
    _, _, test_loader = load_dataset(
        dataset_name=config.DATASET_NAME, batch_size=5, data_dir=config.DATA_DIR
    )
    try:
        transformer.load_state_dict(torch.load(config.TRANSFORMER_MODEL_PATH, map_location=device))
        visualize_colorization_results(
            config, vae_model, transformer, color_quantizer, data_loader=test_loader
        )
    except FileNotFoundError:
        print(f"[WARN] Could not find the best Transformer model at '{config.TRANSFORMER_MODEL_PATH}'.")
