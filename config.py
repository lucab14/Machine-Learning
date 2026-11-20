import torch
import os


class Config:
    """Central configuration with all hyperparameters and file paths used across the project."""

    def __init__(self):
        # General settings
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.OUTPUT_DIR = "./results"  # Where models, checkpoints, plots and images are saved

        # Dataset settings
        self.DATASET_NAME = "places365_standard"  # "places365_standard", "celeba" or "cifar10"
        self.DATA_DIR = "./data"
        self.RESIZE_DIM = 256       # Image resolution used for training and evaluation
        self.SUBSET_SIZE = 200000   # Number of training images to use (None for the full set)

        # U-Net colorization model
        self.COLORIZER_EPOCHS = 100
        self.COLORIZER_BATCH_SIZE = 16
        self.COLORIZER_LEARNING_RATE = 2e-4
        self.COLORIZER_PATIENCE = 10            # Early-stopping patience
        self.COLORIZER_MODEL_PATH = os.path.join(self.OUTPUT_DIR, "best_colorizer_unet_1.pth")
        self.COLORIZER_CHECKPOINT_PATH = os.path.join(self.OUTPUT_DIR, "percep_colorizer_checkpoint_1.pth")
        self.COLORIZER_VIZ_EVERY_N_EPOCHS = 5
        self.L1_LAMBDA = 100.0                  # Weight of the L1 term in the combined loss

        # VAE settings
        self.AE_LATENT_DIM = 512
        self.AE_EPOCHS = 100
        self.AE_LEARNING_RATE = 1e-3
        self.AE_BATCH_SIZE = 32
        self.AE_PATIENCE = 5
        self.AE_MODEL_PATH = os.path.join(self.OUTPUT_DIR, "best_autoencoder_model.pth")
        self.VAE_KL_BETA_FINAL = 0.1            # Final weight of the KL term (beta-VAE)
        self.VAE_KL_WARMUP_EPOCHS = 10          # Epochs over which beta is linearly ramped up

        # Transformer settings
        self.TRANSFORMER_NUM_LAYERS = 8
        self.TRANSFORMER_NHEAD = 8
        self.TRANSFORMER_FF_DIM = 2048
        self.TRANSFORMER_EPOCHS = 50
        self.TRANSFORMER_LEARNING_RATE = 1e-6
        self.TRANSFORMER_BATCH_SIZE = 16
        self.TRANSFORMER_PATIENCE = 5
        self.TRANSFORMER_MODEL_PATH = os.path.join(self.OUTPUT_DIR, "best_transformer_model.pth")
        self.VIZ_EVERY_N_EPOCHS = 2

        # Batch size used to build the data loaders for the VAE and Transformer stages
        self.FULL_MODULE_BATCH_SIZE = 16

        # Ensure the output directory exists
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def print_config(self):
        """Print every configuration attribute and its current value.

        :return: None
        """
        print("--- Configuration Settings ---")
        for attr, value in self.__dict__.items():
            if not attr.startswith('__'):
                print(f"{attr}: {value}")
        print("----------------------------")
