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

    def __init__(self, config_dict):
        """
        Initializes the JepaEncoder using a configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing all necessary parameters.
                                Expected keys are defined in the JSON configuration file.
        """
        if vit is None or AttentiveClassifier is None or load_pretrained is None or load_and_preprocess_video is None:
             raise ImportError("JEPA dependencies (vit, AttentiveClassifier, or utils) not loaded. Cannot initialize JepaEncoder.")

        logger.info("Initializing JepaEncoder from configuration dictionary...")
        params = config_dict.get("parameters")
        if params is None:
            raise ValueError("Configuration dictionary must contain a 'parameters' key.")

        # Determine and set device internally
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"JepaEncoder determined to use device: {self.device}")

        # --- Extract and Store Configuration ---
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Checkpoint paths: make them absolute from project root if relative
        _encoder_ckpt_path_rel = params.get('encoder_checkpoint_path', 'weights/jepa/vit_huge16-384/vith16-384.pth.tar') # Default for safety
        self.encoder_checkpoint_path = os.path.join(_project_root, _encoder_ckpt_path_rel) if not os.path.isabs(_encoder_ckpt_path_rel) else _encoder_ckpt_path_rel

        _probe_ckpt_path_rel = params.get('probe_checkpoint_path', 'weights/jepa/vit_huge16-384/k400-probe.pth.tar') # Default for safety
        self.probe_checkpoint_path = os.path.join(_project_root, _probe_ckpt_path_rel) if not os.path.isabs(_probe_ckpt_path_rel) else _probe_ckpt_path_rel

        if not os.path.isfile(self.encoder_checkpoint_path):
            logger.warning(f"Encoder checkpoint path does not exist or is not a file: {self.encoder_checkpoint_path}")
        if not os.path.isfile(self.probe_checkpoint_path):
            logger.warning(f"Probe checkpoint path does not exist or is not a file: {self.probe_checkpoint_path}")

        self.model_name = params.get('model_name', 'vit_huge')
        self.patch_size = params.get('patch_size', 16)
        self.resolution = params.get('resolution', 384)
        self.frames_per_clip = params.get('frames_per_clip', 16)
        self.tubelet_size = params.get('tubelet_size', 2)
        self.checkpoint_key_encoder = params.get('checkpoint_key_encoder', 'target_encoder')
        self.checkpoint_key_probe = params.get('checkpoint_key_probe', 'classifier')
        self.model_kwargs = params.get('model_kwargs', {
            'use_sdpa': True, 'uniform_power': False, 'use_SiLU': False, 'tight_SiLU': True
        })
        self.norm_mean = params.get('norm_mean', [0.485, 0.456, 0.406])
        self.norm_std = params.get('norm_std', [0.229, 0.224, 0.225])
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
            input_tensor = load_and_preprocess_video(
                video_path=video_path,
                num_frames_to_sample=self.frames_per_clip,
                target_resolution=self.resolution,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                return_format='BCTHW'
            ).to(self.device)

            expected_shape_part = (1, 3, self.frames_per_clip, self.resolution, self.resolution)
            # More robust check for frame count due to short videos
            actual_frames = input_tensor.shape[2]
            if not (input_tensor.shape[0] == 1 and \
                    input_tensor.shape[1] == 3 and \
                    actual_frames <= self.frames_per_clip and \
                    input_tensor.shape[3] == self.resolution and \
                    input_tensor.shape[4] == self.resolution and \
                    actual_frames > 0 ) : # ensure some frames were loaded
                logger.error(f"Video {video_path}: Input tensor shape mismatch or invalid. Expected compatible with {expected_shape_part}, got {input_tensor.shape}. Skipping video.")
                return None
            if actual_frames < self.frames_per_clip:
                 logger.warning(f"Input tensor frame count {actual_frames} is less than config {self.frames_per_clip} (short video). Proceeding.")


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
