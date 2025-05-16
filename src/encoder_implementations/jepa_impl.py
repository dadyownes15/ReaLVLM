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
    from src.jepa.src.models import vision_transformer as vit
    from src.jepa.src.models.attentive_pooler import AttentiveClassifier
    # Util functions from the main src directory
    from src.util import load_pretrained
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
        if vit is None or AttentiveClassifier is None or load_pretrained is None:
             raise ImportError("JEPA dependencies (vit, AttentiveClassifier, or load_pretrained) not loaded.")

        logger.info("Initializing JepaEncoder from configuration dictionary...")
        params = config_dict.get("parameters")
        if params is None:
            raise ValueError("Config missing 'parameters' key.")
        preproc_params = params.get("preprocessing")
        if preproc_params is None:
             raise ValueError("Config missing 'parameters.preprocessing' key.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"JepaEncoder determined to use device: {self.device}")

        # --- Extract and Store Configuration (Strict Checking) ---
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        model_param_keys = [
            'encoder_checkpoint_path', 'probe_checkpoint_path', 'model_name',
            'patch_size', 'model_frames_per_clip', 'tubelet_size',
            'checkpoint_key_encoder', 'checkpoint_key_probe', 'model_kwargs'
        ]
        preproc_keys = [
            'resolution', 'clip_duration_frames', 'clip_stride', 'sampling_method', 
            'resize_method', 'norm_mean', 'norm_std'
        ]
        missing_model_keys = [key for key in model_param_keys if key not in params]
        missing_preproc_keys = [key for key in preproc_keys if key not in preproc_params]
        if missing_model_keys or missing_preproc_keys:
            raise ValueError(f"Missing required config keys. Model params: {missing_model_keys}, Preprocessing params: {missing_preproc_keys}")

        # Store parameters (resolving paths)
        _encoder_ckpt_path_rel = params['encoder_checkpoint_path']
        self.encoder_checkpoint_path = os.path.join(_project_root, _encoder_ckpt_path_rel) if not os.path.isabs(_encoder_ckpt_path_rel) else _encoder_ckpt_path_rel
        _probe_ckpt_path_rel = params['probe_checkpoint_path']
        self.probe_checkpoint_path = os.path.join(_project_root, _probe_ckpt_path_rel) if not os.path.isabs(_probe_ckpt_path_rel) else _probe_ckpt_path_rel

        self.model_name = params['model_name']
        self.patch_size = params['patch_size']
        self.model_frames_per_clip = params['model_frames_per_clip'] # How many frames the model arch takes
        self.tubelet_size = params['tubelet_size']
        self.checkpoint_key_encoder = params['checkpoint_key_encoder']
        self.checkpoint_key_probe = params['checkpoint_key_probe']
        self.model_kwargs = params['model_kwargs']
        
        self.preproc_params = preproc_params
        self.resolution = preproc_params['resolution'] # Convenience reference

        if not os.path.isfile(self.encoder_checkpoint_path):
            logger.warning(f"Encoder checkpoint path not found: {self.encoder_checkpoint_path}")
        if not os.path.isfile(self.probe_checkpoint_path):
            logger.warning(f"Probe checkpoint path not found: {self.probe_checkpoint_path}")

        # --- Initialize Encoder ---
        self.encoder = _init_encoder(
            device=self.device,
            pretrained_path=self.encoder_checkpoint_path,
            checkpoint_key=self.checkpoint_key_encoder,
            model_name=self.model_name,
            patch_size=self.patch_size,
            resolution=self.resolution,
            frames_per_clip=self.model_frames_per_clip,
            tubelet_size=self.tubelet_size,
            model_kwargs=self.model_kwargs
        )

        # --- Initialize Attentive Classifier and Load Probe Weights ---
        logger.info("Initializing Attentive Classifier (Pooler)...")
        try:
            _num_classes = config_dict.get("probe_num_classes", 400) # Default to 400 if not found
            classifier = AttentiveClassifier(
                embed_dim=self.encoder.embed_dim,
                num_heads=self.encoder.num_heads,
                depth=1,
                num_classes=_num_classes
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

    
    def _preprocess_clip(self, clip_frames):
        """(Private) Preprocesses a batch/list of frames for a single clip."""
        # Extract relevant params from self.preproc_params
        resolution = self.preproc_params['resolution']
        norm_mean = self.preproc_params['norm_mean']
        norm_std = self.preproc_params['norm_std']
        resize_method = self.preproc_params['resize_method']

        # Convert frames T H W C (from decord) -> T C H W & normalize [0, 1]
        frames_tensor = torch.from_numpy(clip_frames).permute(0, 3, 1, 2).float() / 255.0
        t, c, h, w = frames_tensor.shape
        if h == 0 or w == 0:
            raise ValueError("Clip frame dimensions are invalid.")
        
        transform_list = []
        # Apply resizing based on method
        if resize_method == 'jepa_short_side':
            if w < h:
                new_w = resolution
                new_h = int(resolution * h / w)
            else:
                new_h = resolution
                new_w = int(resolution * w / h)
            transform_list.append(transforms.Resize((new_h, new_w), antialias=True))
        # Add other resize methods ('hiera_larger_crop', etc.) if needed
        else:
            # Default or fallback resize (e.g., simple resize)
            transform_list.append(transforms.Resize((resolution, resolution), antialias=True))

        transform_list.append(transforms.CenterCrop(resolution))
        transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        transform = transforms.Compose(transform_list)

        # Apply transform
        frames_processed = torch.stack([transform(frame) for frame in frames_tensor]) # Shape: [T, C, H, W]

        # Format for model: B, C, T, H, W 
        # Model's internal patch embed likely handles T->T/tubelet_size
        frames_final = frames_processed.permute(1, 0, 2, 3) # C, T, H, W
        return frames_final.unsqueeze(0) # Add Batch dim -> [1, C, T, H, W]


    @torch.no_grad()
    def encode_video(self, video_path):
        """
        Loads, preprocesses (clip-by-clip), and extracts embeddings for a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[torch.Tensor] or None: A list of PyTorch Tensors, where each tensor
                                        is an embedding for a clip (shape [embed_dim]) on CPU,
                                        or None if an error occurs or no clips are generated.
        """
        logger.info(f"Encoding video (clip-based): {video_path}")
        try:
            vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
            video_len = len(vr)
            if video_len == 0:
                logger.warning(f"Video has 0 frames: {video_path}")
                return None
            
            clip_len = self.preproc_params['clip_duration_frames']
            stride = self.preproc_params['clip_stride']
            # Sampling method might be different ('stride', 'linspace', etc.)
            # For now, implement simple stride sampling
            
            clip_embeddings = []
            # Calculate frame indices for each clip start
            clip_start_indices = list(range(0, video_len - clip_len + 1, stride))
            # Ensure the last few frames are captured if stride doesn't align perfectly
            if (video_len - clip_len) % stride != 0 and (video_len - clip_len) >= 0: 
                 clip_start_indices.append(video_len - clip_len)
                 clip_start_indices = sorted(list(set(clip_start_indices))) # Remove potential duplicate if end aligns
            
            if not clip_start_indices: # Handle very short videos
                 logger.warning(f"Video length ({video_len}) shorter than clip length ({clip_len}). Processing available frames.")
                 # Process the whole video if shorter than a clip
                 frame_indices = np.arange(video_len)
                 frames = vr.get_batch(frame_indices).asnumpy()
                 if frames.shape[0] > 0:
                      logger.warning(f"Processing short video clip of length {frames.shape[0]}")
                      input_tensor = self._preprocess_clip(frames).to(self.device) 
                      encoder_embeddings = self.encoder(input_tensor)
                      pooled_embeddings = self.pooler(encoder_embeddings) # Shape [1, D]
                      clip_embeddings.append(pooled_embeddings.squeeze(0).cpu()) # Store as Tensor
                 else:
                      logger.error(f"Could not read any frames from short video: {video_path}")
                      return None
            else:
                logger.info(f"Processing {len(clip_start_indices)} clips for video: {video_path}")
                for start_idx in clip_start_indices:
                    frame_indices = np.arange(start_idx, start_idx + clip_len)
                    frames = vr.get_batch(frame_indices).asnumpy() # T, H, W, C
                    
                    # Preprocess and encode the clip
                    input_tensor = self._preprocess_clip(frames).to(self.device) # 1, C, T, H, W
                    
                    # Verify input shape matches model expectation (mostly T dim)
                    if input_tensor.shape[2] != self.model_frames_per_clip:
                         logger.error(f"Preprocessed clip frame count ({input_tensor.shape[2]}) doesn't match model expectation ({self.model_frames_per_clip}). Skipping clip.")
                         # This could happen if preprocessing pads/truncates unexpectedly or due to short video logic needs refinement.
                         continue 
                         
                    encoder_embeddings = self.encoder(input_tensor)
                    pooled_embeddings = self.pooler(encoder_embeddings) # Shape [1, D]
                    clip_embeddings.append(pooled_embeddings.squeeze(0).cpu()) # Store as Tensor
                    logger.debug(f"Stored embedding for clip {start_idx}. Original shape: {pooled_embeddings.shape}")

            if not clip_embeddings:
                logger.warning(f"No embeddings generated for video: {video_path}")
                return None

            logger.info(f"Finished encoding video. Generated {len(clip_embeddings)} clip embeddings.")
            return clip_embeddings # Return the list of tensors

        except FileNotFoundError:
            logger.error(f"Video file not found during encoding: {video_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to encode video {video_path}: {e}", exc_info=True)
            return None

# --- Helper Functions --- (Moved below class for clarity)

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
