import torch
import logging
import os
from pathlib import Path
import sys
import numpy as np
from torchvision import transforms
from decord import VideoReader, cpu

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- JEPA and Utility Imports ---
try:
    # Assuming 'jepa' and 'src' are top-level directories relative to the project root
    from jepa.src.models import vision_transformer as vit
    from jepa.src.models.attentive_pooler import AttentiveClassifier
    # Util functions from the main src directory
    from src.util import load_pretrained, load_and_preprocess_video 
except ImportError as e:
    logger.error("Error importing jepa/src modules or util functions. Ensure directories exist and project is run correctly.", exc_info=True)
    exit(1)
# --------------------------------



# --- Main Encoder Class ---
class JepaEncoder:
    """Encapsulates V-JEPA model loading and video embedding extraction."""

    def __init__(self,
                 encoder_checkpoint_path="weights/jepa/vit_huge16-384/vith16-384.pth.tar",
                 probe_checkpoint_path="weights/jepa/vit_huge16-384/k400-probe.pth.tar",
                 device=None,
                 # --- Model Config Parameters with Defaults ---
                 model_name='vit_huge',
                 patch_size=16,
                 resolution=384,
                 frames_per_clip=16,
                 tubelet_size=2,
                 checkpoint_key_encoder='target_encoder',
                 checkpoint_key_probe='classifier',
                 model_kwargs = {
                     'use_sdpa': True,
                     'uniform_power': False,
                     'use_SiLU': False,
                     'tight_SiLU': True,
                 },
                 norm_mean = [0.485, 0.456, 0.406],
                 norm_std = [0.229, 0.224, 0.225]
                 # --- End Model Config Parameters ---
                 ):
        """
        Initializes the JepaEncoder.

        Args:
            encoder_checkpoint_path (str, optional): Path to the V-JEPA pretrained encoder checkpoint.
                                                     Defaults to 'weights/jepa/vit_huge16-384/vith16-384.pth.tar'.
            probe_checkpoint_path (str, optional): Path to the attentive probe checkpoint.
                                                   Defaults to 'weights/jepa/vit_huge16-384/k400-probe.pth.tar'.
            device (str, optional): Device to run the model on ('cuda', 'cpu'). 
                                     Defaults to 'cuda' if available, else 'cpu'.
            model_name (str): Name of the Vision Transformer model variant.
            patch_size (int): Size of patches.
            resolution (int): Input resolution.
            frames_per_clip (int): Number of frames the model expects.
            tubelet_size (int): Temporal size of tubelet patches.
            checkpoint_key_encoder (str): Key for encoder weights in the checkpoint.
            checkpoint_key_probe (str): Key for probe weights in the checkpoint.
            model_kwargs (dict): Additional arguments for the ViT model constructor.
            norm_mean (list): Mean for normalization.
            norm_std (list): Standard deviation for normalization.
        """
        # --- Basic Setup ---
        # Determine project root (assuming this file is in src/encoder_implementations)
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Set default checkpoint paths if None is provided
        if encoder_checkpoint_path is None:
             _default_encoder_path = os.path.join(_project_root, 'weights', 'jepa', 'vit_huge16-384', 'vith16-384.pth.tar')
             if not os.path.isfile(_default_encoder_path):
                 logger.warning(f"Default encoder checkpoint not found at: {_default_encoder_path}")
                 # Optionally raise an error or proceed without a default
                 # raise FileNotFoundError(f"Default encoder checkpoint not found: {_default_encoder_path}")
                 self.encoder_checkpoint_path = None # Set explicitly to None if not found
             else:
                 self.encoder_checkpoint_path = _default_encoder_path
                 logger.info(f"Using default encoder checkpoint: {self.encoder_checkpoint_path}")
        else:
            self.encoder_checkpoint_path = encoder_checkpoint_path
            
        if probe_checkpoint_path is None:
            _default_probe_path = os.path.join(_project_root, 'weights', 'jepa', 'vit_huge16-384', 'k400-probe.pth.tar')
            if not os.path.isfile(_default_probe_path):
                 logger.warning(f"Default probe checkpoint not found at: {_default_probe_path}")
                 # raise FileNotFoundError(f"Default probe checkpoint not found: {_default_probe_path}")
                 self.probe_checkpoint_path = None # Set explicitly to None if not found
            else:
                self.probe_checkpoint_path = _default_probe_path
                logger.info(f"Using default probe checkpoint: {self.probe_checkpoint_path}")
        else:
             self.probe_checkpoint_path = probe_checkpoint_path
        
        # Raise error if paths are still None after checking defaults and none were provided
        if self.encoder_checkpoint_path is None:
             raise ValueError("Encoder checkpoint path must be provided or the default must exist.")
        if self.probe_checkpoint_path is None:
             raise ValueError("Probe checkpoint path must be provided or the default must exist.")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")

        # --- Store Configuration as Instance Attributes ---
        self.model_name = model_name
        self.patch_size = patch_size
        self.resolution = resolution
        self.frames_per_clip = frames_per_clip
        self.tubelet_size = tubelet_size
        self.checkpoint_key_encoder = checkpoint_key_encoder
        self.checkpoint_key_probe = checkpoint_key_probe
        self.model_kwargs = model_kwargs
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        # ------------------------------------------------

        # --- Initialize Encoder ---
        self.encoder = _init_encoder(
            device=self.device,
            pretrained_path=self.encoder_checkpoint_path,
            checkpoint_key=self.checkpoint_key_encoder,
            model_name=self.model_name,
            patch_size=self.patch_size,
            resolution=self.resolution,
            frames_per_clip=self.frames_per_clip,
            tubelet_size=self.tubelet_size,
            model_kwargs=self.model_kwargs
        )

        # --- Initialize Attentive Classifier and Load Probe Weights ---
        logger.info("Initializing Attentive Classifier (Pooler)...")
        try:
            # Placeholder number of classes for pooler initialization
            _num_classes_placeholder = 400 
            classifier = AttentiveClassifier(
                embed_dim=self.encoder.embed_dim,
                num_heads=self.encoder.num_heads,
                depth=1, 
                num_classes=_num_classes_placeholder 
            ).to(self.device)
            
            classifier_loaded = _load_probe_checkpoint(
                classifier=classifier,
                probe_checkpoint_path=self.probe_checkpoint_path,
                probe_checkpoint_key=self.checkpoint_key_probe
            )
            classifier_loaded.eval() 
            self.pooler = classifier_loaded.pooler
            logger.info("Attentive Pooler loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize/load probe from {self.probe_checkpoint_path}: {e}", exc_info=True)
            raise
        
        # --- Store Embedding Dimension --- 
        self.embedding_dim = self.encoder.embed_dim
        logger.info(f"Initialized JepaEncoder. Output embedding dimension: {self.embedding_dim}")


    @torch.no_grad()
    def encode_video(self, video_path):
        """
        Loads, preprocesses, and extracts the pooled embedding for a single video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: The pooled embedding tensor (shape [1, embed_dim]) on CPU, 
                          or None if an error occurs during processing.
        """
        logger.info(f"Encoding video: {video_path}")
        try:
            # Load and preprocess video using the utility function
            # JEPA expects BCTHW format and uses default resize
            input_tensor = load_and_preprocess_video(
                video_path=video_path,
                num_frames_to_sample=self.frames_per_clip,
                target_resolution=self.resolution,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                # frames_per_second=None, # Use default uniform sampling
                # resize_impr='default', # Use default resize
                return_format='BCTHW' # Request Batch dim included
            ).to(self.device)

            # Verify input shape
            expected_shape = (1, 3, self.frames_per_clip, self.resolution, self.resolution)
            if input_tensor.shape != expected_shape:
                # Note: Frame count might differ slightly if video is shorter than frames_per_clip
                # This check assumes the sampling logic always returns exactly frames_per_clip
                # A more robust check might verify C, H, W and that T <= frames_per_clip
                actual_frames = input_tensor.shape[2]
                expected_shape_flexible_t = (1, 3, actual_frames, self.resolution, self.resolution)
                if input_tensor.shape == expected_shape_flexible_t and actual_frames <= self.frames_per_clip:
                     logger.warning(f"Input tensor frame count {actual_frames} differs from config {self.frames_per_clip} (likely short video). Proceeding.")
                else:
                    logger.error(f"Video {video_path}: Input tensor shape mismatch! Expected {expected_shape} or compatible, got {input_tensor.shape}. Skipping video.")
                    return None

            # Perform forward pass
            logger.debug(f"Performing forward pass for {video_path}...")
            encoder_embeddings = self.encoder(input_tensor)
            pooled_embeddings = self.pooler(encoder_embeddings)
            logger.debug(f"Got pooled embeddings shape: {pooled_embeddings.shape} for {video_path}")
            
            return pooled_embeddings.cpu()

        except FileNotFoundError:
            logger.error(f"Video file not found during encoding: {video_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to encode video {video_path}: {e}", exc_info=False)
            logger.debug("Detailed traceback:", exc_info=True) 
            return None



# --- Helper Functions (now accept config as args) ---

def _load_probe_checkpoint(classifier, probe_checkpoint_path, probe_checkpoint_key):
    """Loads the probe checkpoint weights into the AttentiveClassifier."""
    logger.info(f"Loading probe checkpoint from: {probe_checkpoint_path}")
    if not Path(probe_checkpoint_path).is_file():
        logger.error(f"Probe checkpoint file not found at: {probe_checkpoint_path}")
        raise FileNotFoundError(f"Probe checkpoint file not found at: {probe_checkpoint_path}")

    try:
        checkpoint = torch.load(probe_checkpoint_path, map_location='cpu')
        logger.info("Loaded probe checkpoint to CPU.")
    except Exception as e:
        logger.error(f"Failed to load probe checkpoint {probe_checkpoint_path}: {e}", exc_info=True)
        raise

    # Find the probe state dict
    probe_dict = None
    potential_keys = [probe_checkpoint_key, 'state_dict'] # Common keys for probes
    if isinstance(checkpoint, dict):
        for key in potential_keys:
            if key in checkpoint:
                probe_dict = checkpoint[key]
                logger.info(f'Using probe checkpoint key: "{key}"')
                break
        if probe_dict is None:
             # Assume the checkpoint root contains the state dict if keys not found
            probe_dict = checkpoint
            logger.info('No standard key found in probe checkpoint, assuming root is the state_dict.')
    else:
        # Assume the loaded object *is* the state_dict
        probe_dict = checkpoint
        logger.info("Probe checkpoint is not a dictionary, assuming it's the state_dict directly.")


    if probe_dict is None:
        logger.error(f"Could not extract state dictionary from probe checkpoint: {probe_checkpoint_path}")
        raise ValueError(f"Could not extract state dictionary from probe checkpoint: {probe_checkpoint_path}")


    # Clean prefixes if necessary (e.g., if saved with DDP)
    cleaned_probe_dict = {k.replace('module.', ''): v for k, v in probe_dict.items()}

    try:
        msg = classifier.load_state_dict(cleaned_probe_dict, strict=True) # Probes usually require strict loading
        logger.info(f"Loaded probe weights with msg: {msg}")
        if msg.missing_keys or msg.unexpected_keys:
            logger.warning(f"Probe loading finished with missing keys: {msg.missing_keys} or unexpected keys: {msg.unexpected_keys}")
    except Exception as e:
        logger.error(f"Error loading state dict into classifier: {e}", exc_info=True)
        raise

    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        logger.info(f'Probe checkpoint is from epoch: {checkpoint.get("epoch", "N/A")}')

    del checkpoint, probe_dict, cleaned_probe_dict # Free memory
    return classifier

def _init_encoder(device, pretrained_path, checkpoint_key, 
                  model_name, patch_size, resolution, 
                  frames_per_clip, tubelet_size, model_kwargs):
    """Initializes the Vision Transformer model and loads pretrained weights."""
    logger.info(f"Initializing V-JEPA encoder: {model_name}")
    if model_name not in vit.__dict__:
        logger.error(f"Model name '{model_name}' not found in jepa.src.models.vision_transformer")
        raise ValueError(f"Model name '{model_name}' not found in jepa.src.models.vision_transformer")

    try:
        encoder = vit.__dict__[model_name](
            img_size=resolution,
            patch_size=patch_size,
            num_frames=frames_per_clip,
            tubelet_size=tubelet_size,
            **model_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to instantiate model {model_name}: {e}", exc_info=True)
        raise

    # Load pretrained weights using the utility function
    try:
        encoder = load_pretrained(
            encoder=encoder,
            pretrained_path=pretrained_path,
            checkpoint_key=checkpoint_key
        )
    except Exception as e:
         logger.error(f"Failed to load pretrained weights for encoder from {pretrained_path}: {e}", exc_info=True)
         raise

    encoder.to(device)
    encoder.eval() # Set to evaluation mode
    logger.info(f"Encoder moved to {device} and set to eval mode.")
    return encoder
