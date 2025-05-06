import torch
from pathlib import Path
import logging
import numpy as np
from torchvision import transforms
from decord import VideoReader, cpu
import math
import os

# Configure logger for this module
logger = logging.getLogger(__name__)
# Basic configuration if no handlers are present
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define globally accessible video extensions
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

def load_pretrained(encoder, pretrained_path, checkpoint_key='target_encoder'):
    """Loads pretrained weights into the encoder model, handling potential keys and prefixes."""
    logger.info(f'Loading pretrained model from {pretrained_path}')
    if not Path(pretrained_path).is_file():
        logger.error(f"Checkpoint file not found at: {pretrained_path}")
        raise FileNotFoundError(f"Checkpoint file not found at: {pretrained_path}")

    # Load checkpoint to CPU first to avoid GPU memory spikes and allow loading on CPU-only machines
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        logger.info("Loaded checkpoint to CPU.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint file {pretrained_path}: {e}", exc_info=True)
        raise

    # --- Find the state dictionary ---
    state_dict = None
    potential_keys = [checkpoint_key, 'encoder', 'state_dict', 'model_state_dict'] # Common keys
    if isinstance(checkpoint, dict):
        for key in potential_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                logger.info(f'Using checkpoint key: "{key}"')
                break
        if state_dict is None:
            # If no known key found, assume the checkpoint itself is the state_dict
            state_dict = checkpoint
            logger.info('No standard key found, assuming checkpoint root is the state_dict.')
    elif isinstance(checkpoint, (torch.nn.Module, torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        # Handle cases where the entire model object might be saved
        state_dict = checkpoint.state_dict()
        logger.warning("Checkpoint file appears to contain the full model object, extracting state_dict.")
    else:
        # Assume the loaded object is the state_dict directly (e.g., saved via torch.save(model.state_dict(), ...))
        state_dict = checkpoint
        logger.info("Checkpoint file does not appear to be a dictionary or model object, assuming it's the state_dict.")


    if state_dict is None:
         logger.error(f"Could not extract state dictionary from checkpoint: {pretrained_path}")
         raise ValueError(f"Could not extract state dictionary from checkpoint: {pretrained_path}")

    # Remove potential module/backbone prefixes
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        if new_k.startswith('backbone.'):
            new_k = new_k[len('backbone.'):]
        if new_k.startswith('encoder.'): # Add handling for 'encoder.' prefix
             new_k = new_k[len('encoder.'):]
        cleaned_state_dict[new_k] = v

    # Check for shape mismatches and missing/unexpected keys compared to the target model
    model_dict = encoder.state_dict()
    final_dict = {}
    missing_keys = []
    unexpected_keys = list(cleaned_state_dict.keys()) # Start with all keys as potentially unexpected
    shape_mismatches = []

    for k, v in model_dict.items():
        if k in cleaned_state_dict:
            unexpected_keys.remove(k)
            if cleaned_state_dict[k].shape == v.shape:
                final_dict[k] = cleaned_state_dict[k]
            else:
                shape_mismatches.append(f"{k} (model: {v.shape}, ckpt: {cleaned_state_dict[k].shape})")
        else:
            missing_keys.append(k)

    if missing_keys:
         logger.warning(f"Missing keys in target model not found in checkpoint: {missing_keys}")
    if unexpected_keys:
         logger.warning(f"Unexpected keys found in checkpoint not present in target model: {unexpected_keys}")
    if shape_mismatches:
        logger.warning(f"Shape mismatches found (these keys will be skipped): {shape_mismatches}")

    # Load the state dict (non-strictly to allow for mismatches/missing keys)
    msg = encoder.load_state_dict(final_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
         # Log detailed messages only if there were actual issues after filtering
         actual_missing = [k for k in msg.missing_keys if k not in final_dict] # Keys truly not loaded
         actual_unexpected = msg.unexpected_keys # These are always unexpected by strict loading
         if actual_missing:
             logger.warning(f"Final missing keys during load_state_dict: {actual_missing}")
         if actual_unexpected:
              logger.warning(f"Final unexpected keys during load_state_dict: {actual_unexpected}")
         # Also log the initial findings for context
         if missing_keys or unexpected_keys or shape_mismatches:
             logger.info("See previous warnings for details on initially missing/unexpected/mismatched keys.")
    else:
         logger.info("Successfully loaded weights into the model.")

    # Log epoch if available
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
         logger.info(f'Checkpoint is from epoch: {checkpoint["epoch"]}')

    del checkpoint, state_dict, cleaned_state_dict # Free memory
    return encoder

def load_and_preprocess_video(
    video_path,
    num_frames_to_sample,
    target_resolution,
    norm_mean,
    norm_std,
    frames_per_second=None, # Optional: for time-based sampling (like Hiera)
    resize_impr='default', # 'default' or 'hiera' style resize
    return_format='BCTHW' # Output format: 'BCTHW' or 'CTHW'
):
    """Loads a video, samples frames, applies transformations, and returns in specified format."""
    logger.debug(f"Attempting to load video: {video_path}")
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    try:
        # Check extension (though decord might handle it)
        if not str(video_path).lower().endswith(VIDEO_EXTENSIONS):
             logger.warning(f"Processing file with non-standard extension: {video_path}")
             
        vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        video_len = len(vr)
        video_fps = vr.get_avg_fps()
        logger.debug(f"Video loaded: {video_len} frames, {video_fps:.2f} FPS.")

        if video_len == 0:
            raise ValueError(f"Video {video_path} has 0 frames.")

        # --- Frame Sampling Logic ---
        indices = None
        if frames_per_second and video_fps > 0:
             # Time-based sampling (like Hiera)
            total_duration = video_len / video_fps
            sample_interval_time = 1.0 / frames_per_second
            num_actual_samples = min(num_frames_to_sample, math.ceil(total_duration * frames_per_second))
            timestamps = np.linspace(0, total_duration, num=num_actual_samples, endpoint=False)
            indices = np.clip(np.round(timestamps * video_fps), 0, video_len - 1).astype(np.int64)
            indices = np.unique(indices)
            logger.debug(f"Sampling {len(indices)} unique frames via FPS ({frames_per_second}). Indices: {indices.tolist()}")
        else:
            # Uniform frame count sampling (like JEPA)
            if not frames_per_second:
                 logger.debug(f"Using uniform frame sampling ({num_frames_to_sample} frames). Video FPS: {video_fps}")
            else: # FPS was invalid
                 logger.warning(f"Invalid video FPS ({video_fps}). Falling back to uniform frame sampling ({num_frames_to_sample} frames).")
            indices = np.linspace(0, video_len - 1, num=num_frames_to_sample, dtype=np.int64)
            indices = np.clip(indices, 0, video_len - 1)
            logger.debug(f"Sampling {num_frames_to_sample} frames uniformly. Indices: {indices.tolist()}")

        if indices is None or len(indices) == 0:
            logger.warning(f"No frames selected for video {video_path}. Using the middle frame.")
            indices = np.array([video_len // 2], dtype=np.int64)

        frames = vr.get_batch(indices).asnumpy()
        del vr

        # T H W C -> T C H W & normalize [0, 1]
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # --- Transformations ---
        t, c, h, w = frames_tensor.shape
        if h == 0 or w == 0:
            raise ValueError(f"Video {video_path} frame dimensions are invalid: H={h}, W={w}")

        # Choose resize strategy
        if resize_impr == 'hiera':
             # Resize slightly larger then center crop (common for ImageNet models)
             resize_size = target_resolution + 32 # Example heuristic, adjust if needed
             transform_list = [
                 transforms.Resize(resize_size),
                 transforms.CenterCrop(target_resolution),
             ]
        else: # Default JEPA-style resize short side
             if w < h:
                 new_w = target_resolution
                 new_h = int(target_resolution * h / w)
             else:
                 new_h = target_resolution
                 new_w = int(target_resolution * w / h)
             transform_list = [
                  transforms.Resize((new_h, new_w), antialias=True),
                  transforms.CenterCrop(target_resolution),
             ]

        # Add normalization
        transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        transform = transforms.Compose(transform_list)

        # Apply transform
        frames_processed = torch.stack([transform(frame) for frame in frames_tensor]) # Shape: [T, C, H, W]

        # --- Format Output --- 
        if return_format == 'CTHW':
             frames_final = frames_processed.permute(1, 0, 2, 3) # C, T, H, W
             logger.debug(f"Returning preprocessed video in CTHW format: {frames_final.shape}")
             return frames_final
        elif return_format == 'BCTHW':
             frames_final = frames_processed.permute(1, 0, 2, 3) # C, T, H, W
             frames_final = frames_final.unsqueeze(0) # B, C, T, H, W
             logger.debug(f"Returning preprocessed video in BCTHW format: {frames_final.shape}")
             return frames_final
        else: # Default to TCHW if format unknown
             logger.warning(f"Unknown return format '{return_format}'. Defaulting to TCHW.")
             logger.debug(f"Returning preprocessed video in TCHW format: {frames_processed.shape}")
             return frames_processed

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
        # Raise the exception so the caller knows processing failed
        raise




def load_embeddings_from_dir(directory_path):
    embeddings_list = []
    logger.info(f"Loading embeddings from: {directory_path}")
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return None
        
    for filename in os.listdir(directory_path):
        if filename.endswith(".pt"): # Or .npy if you change saving format
            file_path = os.path.join(directory_path, filename)
            try:
                # Load tensor, move to CPU if it's on GPU, then to NumPy
                embedding_tensor = torch.load(file_path, map_location='cpu')
                
                # Ensure it's a 1D array (embedding_dim,) or handle [1, embedding_dim]
                if embedding_tensor.ndim > 1:
                    embedding_tensor = embedding_tensor.squeeze() # Remove batch dim if present
                
                embeddings_list.append(embedding_tensor.numpy())
                logger.debug(f"Loaded and processed {filename}, shape: {embedding_tensor.numpy().shape}")
            except Exception as e:
                logger.error(f"Failed to load or process {file_path}: {e}")
    
    if not embeddings_list:
        logger.warning(f"No embeddings loaded from {directory_path}")
        return None
        
    # Stack into a single NumPy array
    all_embeddings = np.vstack(embeddings_list)
    logger.info(f"Successfully loaded {all_embeddings.shape[0]} embeddings with dimension {all_embeddings.shape[1]}.")
    return all_embeddings


