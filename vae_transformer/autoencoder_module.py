import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Bootstrap: make the shared modules (config, utils) importable from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_dataset, RESIZE_DIM
from config import Config


def weights_init(m):
    """Apply Kaiming (He) initialization to conv/linear layers and identity-like init to BatchNorm.

    :param m: module to initialize
    :return: None
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # He initialization, optimal for layers followed by ReLU.
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm so it acts as identity at the first step.
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class VAE_EncoderUNet(nn.Module):
    """Convolutional VAE encoder producing the latent Gaussian parameters (mu, logvar)."""

    def __init__(self, latent_dim=512, base_channels=32):
        """
        :param latent_dim: dimensionality of the latent space
        :param base_channels: number of channels in the first convolutional block
        """
        super(VAE_EncoderUNet, self).__init__()
        ch = base_channels

        # Each block halves the spatial size and doubles the number of channels.
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=4, stride=2, padding=1),        # 256 -> 128
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1),     # 128 -> 64
            nn.BatchNorm2d(ch*2), nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.BatchNorm2d(ch*4), nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2, padding=1),   # 32 -> 16
            nn.BatchNorm2d(ch*8), nn.ReLU(inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(ch*8, ch*16, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(ch*16), nn.ReLU(inplace=True)
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(ch*16, ch*32, kernel_size=4, stride=2, padding=1), # 8 -> 4
            nn.BatchNorm2d(ch*32), nn.ReLU(inplace=True)
        )

        # Flatten and project to the latent distribution parameters.
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear((ch*32)*4*4, latent_dim)      # mean vector
        self.fc_logvar = nn.Linear((ch*32)*4*4, latent_dim)  # log-variance vector

    def forward(self, x):
        """
        :param x: input grayscale image, shape (N, 1, H, W)
        :return: tuple (mu, logvar, skip_connections)
        """
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        skip5 = self.enc5(skip4)
        out6 = self.enc6(skip5)

        flat = self.flatten(out6)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, (skip1, skip2, skip3, skip4, skip5, out6)


class VAE_Decoder_NoSkips(nn.Module):
    """Convolutional VAE decoder that reconstructs the image from a latent vector z (no skip connections)."""

    def __init__(self, latent_dim=512, base_channels=32):
        """
        :param latent_dim: dimensionality of the latent space
        :param base_channels: number of channels matching the encoder
        """
        super(VAE_Decoder_NoSkips, self).__init__()
        ch = base_channels

        # Project the latent vector back to the initial feature-map size.
        self.fc_decode = nn.Linear(latent_dim, (ch*32)*4*4)
        self.unflatten = nn.Unflatten(1, (ch*32, 4, 4))

        # Each block doubles the spatial size and halves the number of channels.
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(ch*32, ch*16, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(ch*16), nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(ch*16, ch*8, kernel_size=4, stride=2, padding=1),   # 8 -> 16
            nn.BatchNorm2d(ch*8), nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=4, stride=2, padding=1),    # 16 -> 32
            nn.BatchNorm2d(ch*4), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=4, stride=2, padding=1),    # 32 -> 64
            nn.BatchNorm2d(ch*2), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ch*2, ch, kernel_size=4, stride=2, padding=1),      # 64 -> 128
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1),        # 128 -> 256
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        # Final conv to a single channel, with Sigmoid mapping the output to [0, 1].
        self.final_conv = nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, z):
        """
        :param z: latent vector sampled from the latent space, shape (N, latent_dim)
        :return: reconstructed image in [0, 1], shape (N, 1, H, W)
        """
        x = self.fc_decode(z)
        x = self.unflatten(x)
        x = self.dec6(x)
        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.final_conv(x)
        return self.output_act(x)


class VAE_NoSkips(nn.Module):
    """Full VAE combining encoder and decoder, with the reparameterization trick."""

    def __init__(self, latent_dim=512, base_channels=32):
        """
        :param latent_dim: dimensionality of the latent space
        :param base_channels: number of channels in the first block
        """
        super(VAE_NoSkips, self).__init__()
        self.encoder = VAE_EncoderUNet(latent_dim, base_channels)
        self.decoder = VAE_Decoder_NoSkips(latent_dim, base_channels)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + eps * std with eps ~ N(0, 1), keeping sampling differentiable.

        :param mu: mean of the latent distribution
        :param logvar: log-variance of the latent distribution
        :return: sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        :param x: input image, shape (N, 1, H, W)
        :return: tuple (reconstruction, mu, logvar)
        """
        mu, logvar, _ = self.encoder(x)  # skip connections are ignored here
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_unet_loss(recon_x, x, mu, logvar, kl_weight=1.0):
    """Compute the VAE loss (ELBO): reconstruction MSE plus weighted KL divergence.

    :param recon_x: reconstructed image
    :param x: original image
    :param mu: mean of the latent distribution
    :param logvar: log-variance of the latent distribution
    :param kl_weight: weight applied to the KL term
    :return: tuple (total_loss, recon_loss, kl_div)
    """
    B = x.shape[0]

    # Reconstruction term: per-image mean squared error (sum / batch size).
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / B

    # KL term: closed-form divergence between the learned distribution and the standard Gaussian prior.
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    total_loss = recon_loss + kl_weight * kl_div
    return total_loss, recon_loss, kl_div


def train_vae_unet(model, train_loader, val_loader, epochs, learning_rate, device, config):
    """Train the VAE with a beta warm-up schedule and early stopping.

    :param model: the VAE model
    :param train_loader: training DataLoader yielding (grayscale, _) batches
    :param val_loader: validation DataLoader
    :param epochs: maximum number of epochs
    :param learning_rate: optimizer learning rate
    :param device: device the model runs on
    :param config: Config object with the KL warm-up and checkpoint settings
    :return: tuple (trained model, loss history dict)
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config.AE_PATIENCE

    history = {'train_total_loss': [], 'train_recon_loss': [], 'train_kl_loss': [],
               'val_total_loss': [], 'val_recon_loss': [], 'val_kl_loss': []}

    for epoch in range(epochs):
        model.train()
        running_total, running_recon, running_kl = 0.0, 0.0, 0.0

        # beta-VAE warm-up: linearly ramp the KL weight over the first epochs to stabilize training.
        beta_current = min(1.0, (epoch + 1) / config.VAE_KL_WARMUP_EPOCHS) * config.VAE_KL_BETA_FINAL

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]  beta={beta_current:.1e}")
        for (grayscale_img, _color_img) in loop:
            # Map the input from [-1, 1] to [0, 1] to match the decoder's Sigmoid output.
            x = (grayscale_img.to(device) * 0.5) + 0.5

            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            total_loss, recon_loss, kl_div = vae_unet_loss(recon, x, mu, logvar, kl_weight=beta_current)

            total_loss.backward()
            # Gradient clipping to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_total += total_loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_div.item()
            loop.set_postfix({
                "TotLoss": f"{total_loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_div.item():.4f}"
            })

        avg_train_total = running_total / len(train_loader)
        avg_train_recon = running_recon / len(train_loader)
        avg_train_kl = running_kl / len(train_loader)
        history['train_total_loss'].append(avg_train_total)
        history['train_recon_loss'].append(avg_train_recon)
        history['train_kl_loss'].append(avg_train_kl)

        # Validation loop.
        model.eval()
        val_running_total, val_running_recon, val_running_kl = 0.0, 0.0, 0.0
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]  beta={beta_current:.1e}")
            for (grayscale_img, _color_img) in loop_val:
                x_val = (grayscale_img.to(device) * 0.5) + 0.5
                recon_val, mu_val, logvar_val = model(x_val)
                total_loss_val, recon_loss_val, kl_div_val = vae_unet_loss(recon_val, x_val, mu_val, logvar_val, kl_weight=beta_current)

                val_running_total += total_loss_val.item()
                val_running_recon += recon_loss_val.item()
                val_running_kl += kl_div_val.item()
                loop_val.set_postfix({
                    "TotLoss": f"{total_loss_val.item():.4f}",
                    "Recon": f"{recon_loss_val.item():.4f}",
                    "KL": f"{kl_div_val.item():.4f}"
                })

        avg_val_total = val_running_total / len(val_loader)
        avg_val_recon = val_running_recon / len(val_loader)
        avg_val_kl = val_running_kl / len(val_loader)
        history['val_total_loss'].append(avg_val_total)
        history['val_recon_loss'].append(avg_val_recon)
        history['val_kl_loss'].append(avg_val_kl)

        print(f"\n-> Epoch {epoch+1}/{epochs}: "
              f"Train Total={avg_train_total:.4f} (Recon={avg_train_recon:.4f}, KL={avg_train_kl:.4f}) | "
              f"Val Total={avg_val_total:.4f} (Recon={avg_val_recon:.4f}, KL={avg_val_kl:.4f})\n")

        # Reset the best loss at the end of the warm-up, since the loss scale changes.
        if epoch == config.VAE_KL_WARMUP_EPOCHS - 1:
            print("\n  -> Warm-up finished. Establishing a new baseline for the best loss.")
            best_val_loss = avg_val_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.AE_MODEL_PATH)
            print(f"  -> [Baseline Checkpoint] New best VAE saved (Val Loss: {best_val_loss:.4f})")
        else:
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                epochs_no_improve = 0
                torch.save(model.state_dict(), config.AE_MODEL_PATH)
                print(f"  -> [Checkpoint] Best VAE saved (Val Loss: {best_val_loss:.4f})")
            else:
                # Early stopping only counts after the warm-up phase.
                if epoch >= config.VAE_KL_WARMUP_EPOCHS:
                    epochs_no_improve += 1
                    print(f"  -> No validation improvement. Patience: {epochs_no_improve}/{patience}")

        if epoch >= config.VAE_KL_WARMUP_EPOCHS and epochs_no_improve >= patience:
            print(f"\n  -> Early stopping: no improvement for {patience} epochs after warm-up.")
            break

    return model, history


def visualize_reconstruction(model, data_loader, device, num_images=5, save_path="reconstructions.png"):
    """Save a comparison grid of original images and their VAE reconstructions.

    :param model: trained VAE model
    :param data_loader: loader providing (grayscale, _) batches
    :param device: device the model runs on
    :param num_images: number of images to display
    :param save_path: path where the figure is saved
    :return: None
    """
    model.eval()
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2.5, 5))
    fig.suptitle("Originals vs. VAE Reconstructions", fontsize=16)

    with torch.no_grad():
        grayscale_img, _ = next(iter(data_loader))
        x = (grayscale_img[:num_images].to(device) * 0.5) + 0.5  # take num_images and normalize to [0, 1]
        reconstructed_img, _, _ = model(x)

        for i in range(num_images):
            axs[0, i].imshow(x[i, 0].cpu().numpy(), cmap='gray')
            axs[0, i].set_title("Original")
            axs[0, i].axis('off')

            axs[1, i].imshow(reconstructed_img[i, 0].cpu().numpy(), cmap='gray')
            axs[1, i].set_title("Reconstructed")
            axs[1, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Reconstruction image saved to: {save_path}")


if __name__ == "__main__":
    # Entry point: build the VAE, train it, then visualize reconstructions on the test set.
    config = Config()
    device = config.DEVICE
    print(f"[INFO] Selected device: {device}")

    vae_model = VAE_NoSkips(latent_dim=config.AE_LATENT_DIM, base_channels=32).to(device)
    vae_model.apply(weights_init)
    print(f"[INFO] VAE architecture built with {config.AE_LATENT_DIM} latent dimensions.")

    train_loader, val_loader, test_loader = load_dataset(
        dataset_name=config.DATASET_NAME,
        batch_size=config.FULL_MODULE_BATCH_SIZE,
        subset_size=config.SUBSET_SIZE,
        data_dir=config.DATA_DIR
    )

    print("\n[INFO] Starting VAE training...")
    trained_vae, history = train_vae_unet(
        model=vae_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.AE_EPOCHS,
        learning_rate=config.AE_LEARNING_RATE,
        device=device,
        config=config
    )

    # Load the best saved model for the final visualization.
    try:
        trained_vae.load_state_dict(torch.load(config.AE_MODEL_PATH, map_location=device))
        print(f"[INFO] Best VAE model loaded from: {config.AE_MODEL_PATH}")
    except FileNotFoundError:
        print(f"[WARN] Model file '{config.AE_MODEL_PATH}' not found. Using the just-trained model.")

    print("[INFO] Generating final reconstruction images...")
    visualize_reconstruction(
        model=trained_vae,
        data_loader=test_loader,
        device=device,
        num_images=8,
        save_path=os.path.join(config.OUTPUT_DIR, "vae_final_reconstructions.png")
    )
    print("[INFO] Done.")
