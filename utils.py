import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import os
import ssl

# On some systems (e.g. macOS) torchvision dataset downloads fail due to SSL certificate
# verification. Disabling it allows the automatic download; not needed if data is placed manually.
ssl._create_default_https_context = ssl._create_unverified_context

from config import Config
config = Config()

# Target resolution: every image is resized to this size to guarantee uniform inputs.
RESIZE_DIM = config.RESIZE_DIM


class ColorizationTransforms:
    """Converts a PIL RGB image to Lab and splits it into the L input and the a*b* target, both in [-1, 1]."""

    def __init__(self, resize_dim):
        """
        :param resize_dim: side length the image is resized to
        """
        self.resize_dim = resize_dim

    def __call__(self, img_pil):
        """
        :param img_pil: input PIL RGB image
        :return: tuple (L_channel, ab_channels), each a float tensor normalized to [-1, 1]
        """
        # Resize with bicubic interpolation, then convert RGB -> CIELAB.
        img_pil_resized = img_pil.resize((self.resize_dim, self.resize_dim), Image.BICUBIC)
        img_lab = rgb2lab(np.array(img_pil_resized))

        # (H, W, C) -> (C, H, W) tensor, then split luminance and chrominance.
        img_lab_tensor = torch.from_numpy(img_lab.transpose(2, 0, 1)).float()
        L_channel = img_lab_tensor[0:1, :, :]
        ab_channels = img_lab_tensor[1:3, :, :]

        # Normalize to [-1, 1]: L has range [0, 100], a*b* have range ~[-128, 127].
        L_normalized = (L_channel / 50.0) - 1.0
        ab_normalized = ab_channels / 128.0
        return L_normalized, ab_normalized


class CelebaCustomDataset(Dataset):
    """Reads CelebA images directly from a folder, without the official partition files."""

    def __init__(self, img_dir, transform):
        """
        :param img_dir: folder containing the images
        :param transform: callable applied to each loaded PIL image
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(self.image_filenames)} images in {self.img_dir}")

    def __len__(self):
        """:return: number of images in the dataset"""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        :param idx: image index
        :return: the transformed image as an (L, ab) tuple
        """
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        img_pil_original = Image.open(img_path).convert('RGB')
        return self.transform(img_pil_original)


class PlacesFlatDataset(Dataset):
    """Dataset for the flat Places365 validation/test folders (no class subdirectories)."""

    def __init__(self, img_dir, transform):
        """
        :param img_dir: folder containing the images
        :param transform: callable applied to each loaded PIL image
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(self.image_filenames)} images in {self.img_dir}")

    def __len__(self):
        """:return: number of images in the dataset"""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        :param idx: image index
        :return: the transformed image as an (L, ab) tuple
        """
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        img_pil_original = Image.open(img_path).convert('RGB')
        return self.transform(img_pil_original)


class ImageFolderWrapper(Dataset):
    """Wraps torchvision ImageFolder to drop the class label and return only the (L, ab) image data."""

    def __init__(self, image_folder_dataset):
        """
        :param image_folder_dataset: an ImageFolder instance
        """
        self.image_folder_dataset = image_folder_dataset

    def __len__(self):
        """:return: number of images in the dataset"""
        return len(self.image_folder_dataset)

    def __getitem__(self, idx):
        """
        :param idx: image index
        :return: the image data only (label discarded)
        """
        image_data, _ = self.image_folder_dataset[idx]
        return image_data


def load_dataset(dataset_name="celeba", data_dir="./data", batch_size=32, subset_size=None):
    """Build the train/validation/test DataLoaders for the requested dataset.

    :param dataset_name: one of "cifar10", "celeba" or "places365_standard"
    :param data_dir: root directory containing the dataset
    :param batch_size: batch size for all loaders
    :param subset_size: if set, use at most this many images (useful for quick runs)
    :return: tuple (train_loader, val_loader, test_loader)
    """
    colorization_transforms = ColorizationTransforms(RESIZE_DIM)

    train_dataset_final = None
    val_dataset_final = None
    test_dataset_wrapped = None

    # Use a number of workers proportional to the available CPU cores.
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 0

    if dataset_name == "cifar10":
        print(f"Loading CIFAR-10 from {data_dir} (upsampled to {RESIZE_DIM}x{RESIZE_DIM})...")

        class CIFAR10Wrapper(Dataset):
            """Adapts CIFAR-10 output to the (L, ab) format by discarding the label."""

            def __init__(self, cifar_dataset, transform):
                self.cifar_dataset = cifar_dataset
                self.transform = transform

            def __len__(self):
                return len(self.cifar_dataset)

            def __getitem__(self, idx):
                img_pil, _ = self.cifar_dataset[idx]
                return self.transform(img_pil)

        train_dataset_raw = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        test_dataset_raw = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)

        train_dataset_temp = CIFAR10Wrapper(train_dataset_raw, colorization_transforms)
        test_dataset_wrapped = CIFAR10Wrapper(test_dataset_raw, colorization_transforms)

        # 90/10 split of the training set to build a validation set.
        train_size = int(0.9 * len(train_dataset_temp))
        val_size = len(train_dataset_temp) - train_size
        train_dataset_final, val_dataset_final = random_split(train_dataset_temp, [train_size, val_size])

    elif dataset_name == "celeba":
        print(f"Loading CelebA from local folder (resized to {RESIZE_DIM}x{RESIZE_DIM})...")
        celeba_img_dir = os.path.join(data_dir, 'img_align_celeba')
        if not os.path.exists(celeba_img_dir):
            raise FileNotFoundError(f"Folder 'img_align_celeba' not found in {data_dir}.")

        full_dataset_raw = CelebaCustomDataset(celeba_img_dir, colorization_transforms)

        # Optionally take a random subset for quick experiments.
        if subset_size is not None and subset_size < len(full_dataset_raw):
            print(f"Creating a CelebA subset of {subset_size} images.")
            indices = torch.randperm(len(full_dataset_raw))[:subset_size]
            base_dataset = Subset(full_dataset_raw, indices)
        else:
            base_dataset = full_dataset_raw

        # Hierarchical split: 80/20 (train_val / test), then 90/10 (train / val).
        train_val_size = int(0.8 * len(base_dataset))
        test_size = len(base_dataset) - train_val_size
        train_val_dataset, test_dataset_wrapped = random_split(base_dataset, [train_val_size, test_size])

        train_size = int(0.9 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset_final, val_dataset_final = random_split(train_val_dataset, [train_size, val_size])

    elif dataset_name == "places365_standard":
        print(f"Loading Places365-Standard (resized to {RESIZE_DIM}x{RESIZE_DIM})...")
        train_dir = os.path.join(data_dir, 'data_large')
        val_dir = os.path.join(data_dir, 'val_large')
        test_dir = os.path.join(data_dir, 'test_large')
        if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
            raise FileNotFoundError("One or more Places365 folders were not found.")

        # The training set has class subfolders, so ImageFolderWrapper is used.
        train_dataset_raw = ImageFolderWrapper(datasets.ImageFolder(root=train_dir, transform=colorization_transforms))

        # The validation and test sets are flat folders.
        val_dataset_final = PlacesFlatDataset(img_dir=val_dir, transform=colorization_transforms)
        test_dataset_wrapped = PlacesFlatDataset(img_dir=test_dir, transform=colorization_transforms)

        if subset_size is not None and subset_size < len(train_dataset_raw):
            print(f"Using a training subset of {subset_size} images.")
            train_dataset_final = Subset(train_dataset_raw, torch.randperm(len(train_dataset_raw))[:subset_size])
        else:
            train_dataset_final = train_dataset_raw

    else:
        raise ValueError("Unsupported dataset. Choose 'cifar10', 'celeba' or 'places365_standard'.")

    # pin_memory=True speeds up host-to-GPU transfers.
    train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset '{dataset_name}' loaded.")
    print(f"Training set size: {len(train_dataset_final)}")
    print(f"Validation set size: {len(val_dataset_final)}")
    print(f"Test set size: {len(test_dataset_wrapped)}")
    return train_loader, val_loader, test_loader


def visualize_lab_results(model, data_loader, device, epoch, save_dir, num_images=5):
    """Save a comparison grid of input L channel, colorized output and ground truth.

    :param model: model predicting a*b* from the L channel
    :param data_loader: loader providing (L, ab) batches
    :param device: device the model runs on
    :param epoch: current epoch index (used in the title and filename)
    :param save_dir: directory where the figure is saved
    :param num_images: number of images to display
    :return: None
    """
    model.eval()
    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    fig.suptitle(f"Results at epoch: {epoch+1}", fontsize=16)

    with torch.no_grad():
        L_inputs, ab_targets = next(iter(data_loader))
        L_inputs = L_inputs.to(device)[:num_images]
        ab_targets = ab_targets.to(device)[:num_images]

        predicted_ab = model(L_inputs)

        for i in range(num_images):
            # Denormalize back to the original Lab ranges.
            L_channel = (L_inputs[i].cpu() + 1.0) * 50.0
            target_ab = ab_targets[i].cpu() * 128.0
            predicted_ab_denorm = predicted_ab[i].cpu() * 128.0

            # Recombine channels into full Lab images, then convert to RGB for display.
            original_lab = torch.cat([L_channel, target_ab], dim=0).numpy().transpose(1, 2, 0)
            predicted_lab = torch.cat([L_channel, predicted_ab_denorm], dim=0).numpy().transpose(1, 2, 0)
            original_rgb = lab2rgb(original_lab)
            predicted_rgb = lab2rgb(predicted_lab)

            axs[0, i].imshow(L_channel.squeeze(), cmap='gray'); axs[0, i].set_title("Input (Grayscale L)"); axs[0, i].axis("off")
            axs[1, i].imshow(np.clip(predicted_rgb, 0, 1)); axs[1, i].set_title("Colorized Output"); axs[1, i].axis("off")
            axs[2, i].imshow(np.clip(original_rgb, 0, 1)); axs[2, i].set_title("Ground Truth"); axs[2, i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"Perc_1_colorization_epoch_{epoch+1:03d}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Visualization saved to: {save_path}")


if __name__ == "__main__":
    # Smoke test: load the dataset and print the shape of the first batch.
    print("Testing with Places365-Standard...")
    try:
        train_loader_places, val_loader_places, test_loader_places = load_dataset(
            dataset_name="places365_standard",
            batch_size=config.FULL_MODULE_BATCH_SIZE,
            subset_size=config.SUBSET_SIZE,
            data_dir=config.DATA_DIR
        )
        for i, (L_channel, ab_channels) in enumerate(train_loader_places):
            print(f"Batch {i}: L shape: {L_channel.shape}, ab shape: {ab_channels.shape}")
            if i == 0:
                break
    except Exception as e:
        print(f"Error while loading Places365-Standard: {e}")
        print("Make sure the dataset folders are structured correctly.")
