import torch
from pathlib import Path
import logging
import numpy as np
# from torchvision import transforms # No longer needed here if preprocess in encoders
# from decord import VideoReader, cpu # No longer needed here
# import math # No longer needed here
import os

# Configure logger for this module
logger = logging.getLogger(__name__)
# Basic configuration if no handlers are present
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define globally accessible video extensions (can be used by EncoderModel)
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

# -- Removed load_and_preprocess_video function --
# Preprocessing logic is now expected within each encoder implementation's
# encode_video method or private helper methods, driven by their config.

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


