# experiment_config.py

# Configuration settings for the VQ-UNET experiment
import torch
class config:
    # Dataset parameters
    DATA_ROOT = './data'  # Root directory for datasets (PASCAL VOC will be downloaded here)
    BATCH_SIZE = 16      # Batch size for training (adjusted for potentially larger model)
    IMAGE_SIZE = (128, 128) # Image size for input
    
    # Model parameters
    # For VQUNet:
    # ENCODER_CHANNEL_DIMS will define the embedding dimensions for each codebook
    # e.g., [64, 128, 256, 512] for a 4-stage encoder + bottleneck
    CODEBOOK_SIZE = 512                   # Number of vectors in each codebook
    COMMITMENT_COST = 0.25                # Commitment cost for VQ loss
    # NUM_CLASSES is not directly used for reconstruction model, but can be kept if other tasks use it.
    # LATENT_DIM, EMBEDDING_DIM, SKIP_CONNECTIONS are superseded by new VQUNet design.

    # Training parameters
    LEARNING_RATE = 1e-4                  # Learning rate for the optimizer
    NUM_EPOCHS = 25                       # Number of training epochs (adjust as needed)
    WEIGHT_DECAY = 1e-5                   # Weight decay for regularization
    VQ_LOSS_WEIGHT = 1.0                  # Weight for the VQ loss component

    # Logging parameters
    LOG_INTERVAL = 10                     # Interval for logging training progress
    CHECKPOINT_DIR = 'checkpoints/'       # Directory to save model checkpoints

    # Device configuration
    USE_CUDA = True                       # Use CUDA if available
    DEVICE = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'  # Device to run the model on

    # Additional configurations
    SEED = 42                             # Random seed for reproducibility
    # DATA_AUGMENTATION can be added to transforms if needed

    @staticmethod
    def display():
    
        print("Experiment Configuration:")
        # Update DEVICE based on actual availability for display
        current_device = 'cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu'
        attrs = {key: value for key, value in vars(config).items() if not key.startswith('__') and not callable(value)}
        attrs['DEVICE'] = current_device # Ensure displayed DEVICE is accurate
        for key, value in attrs.items():
            print(f"{key}: {value}")


config.DEVICE = 'cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu'