# InternVideo2s1-1b

import torch
import logging
import os
from pathlib import Path
import sys
import numpy as np
from torchvision import transforms
from decord import VideoReader, cpu
import math

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import transformers
try:
    from transformers import AutoModel, AutoConfig, AutoProcessor
except ImportError:
    logger.error("transformers library not found. Please install it (e.g., pip install transformers)")
    AutoModel = None
    AutoConfig = None
    AutoProcessor = None
    # exit(1)

# --- Constants for InternVideo2 ---
# Default local path to the *directory* containing config and weights
DEFAULT_MODEL_PATH = "weights/internvideo"
# Explicit name of the non-standard weight file within that directory
WEIGHT_FILENAME = "internvideo_dist_L.bin" 
# Settings inferred from model name or typical defaults - verify with model card/processor if possible
NUM_FRAMES_TO_SAMPLE = 8 # Example, adjust if needed
RESOLUTION = 224 # Example, adjust if needed
BATCH_SIZE = 1
# --------------------------------

def _load_and_preprocess_video_internvideo(
    video_path,
    processor,
    num_frames_to_sample=NUM_FRAMES_TO_SAMPLE
):
    """Loads a video, samples frames uniformly, and uses the Hugging Face processor for transformations."""
    logger.debug(f"Attempting to load video: {video_path}")
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    try:
        vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        video_len = len(vr)
        logger.debug(f"Video loaded: {video_len} frames.")

        if video_len == 0:
            raise ValueError(f"Video {video_path} has 0 frames.")

        # --- Frame Sampling Logic (Uniform Sampling) ---
        indices = np.linspace(0, video_len - 1, num=num_frames_to_sample, dtype=np.int64)
        indices = np.clip(indices, 0, video_len - 1)
        logger.debug(f"Sampling {num_frames_to_sample} frames at indices: {indices.tolist()}")
        
        # Get frames [T, H, W, C]
        frames = vr.get_batch(indices).asnumpy()
        del vr
        
        # Convert frames to list of PIL images or numpy arrays for the processor
        # Processor expects a list of frames (e.g., List[np.ndarray] or List[PIL.Image])
        # Assuming processor handles channel order and normalization
        # Input shape for processor: List of [H, W, C] numpy arrays
        frames_list = [frame for frame in frames]

        # --- Use Processor for Transformations ---
        # The processor should handle resizing, cropping, normalization, and tensor conversion
        # It typically returns a dictionary including 'pixel_values'
        inputs = processor(images=frames_list, return_tensors="pt")
        
        # Expected output shape from processor might be [1, T, C, H, W] or similar
        logger.debug(f"Processor output keys: {inputs.keys()}")
        if 'pixel_values' not in inputs:
             raise ValueError("Processor did not return 'pixel_values'. Check processor usage.")
             
        logger.debug(f"Processor output pixel_values shape: {inputs['pixel_values'].shape}")
        # Return the pixel values tensor
        return inputs['pixel_values'] # Shape might be e.g., [1, T, C, H, W] or [B, C, T, H, W]

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
        raise


class InternVideoEncoder:
    """Encapsulates InternVideo2 model loading and video embedding extraction."""

    def __init__(self, model_path_or_id=None, device=None, weight_filename=None):
        """
        Initializes the InternVideoEncoder.

        Args:
            model_path_or_id (str, optional): Path to the local directory containing the model config/processor 
                                              files OR a Hugging Face model ID.
                                              Defaults to 'weights/internvideo'.
            device (str, optional): Device to run the model on ('cuda', 'cpu'). 
                                     Defaults to 'cuda' if available, else 'cpu'.
            weight_filename (str, optional): Explicit filename for the weight file (e.g., 'pytorch_model.bin', 
                                             'internvideo_dist_L.bin') if loading locally and it's non-standard.
                                             Defaults to WEIGHT_FILENAME ('internvideo_dist_L.bin').
        """
        if AutoModel is None or AutoProcessor is None or AutoConfig is None:
            raise ImportError("transformers library AutoModel/AutoProcessor/AutoConfig is not available. Cannot initialize InternVideoEncoder.")

        # --- Determine Model Path ---
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if model_path_or_id is None:
            self.model_path = os.path.join(_project_root, DEFAULT_MODEL_PATH)
            logger.info(f"Using default local model directory: {self.model_path}")
            self.is_local = True
        else:
            if os.path.isdir(model_path_or_id):
                self.model_path = model_path_or_id
                logger.info(f"Using provided local model directory: {self.model_path}")
                self.is_local = True
            else:
                 self.model_path = model_path_or_id
                 logger.info(f"Using provided Hugging Face model ID: {self.model_path}")
                 self.is_local = False
                 
        if self.is_local and not os.path.isdir(self.model_path):
             raise FileNotFoundError(f"Specified local model directory does not exist: {self.model_path}")
             
        # Determine weight filename for local loading
        self.weight_filename = weight_filename if weight_filename is not None else WEIGHT_FILENAME
        # -------------------------
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing InternVideoEncoder: source='{self.model_path}' on device '{self.device}'")

        # --- Load Model and Processor --- 
        try:
            logger.info(f"Loading processor from '{self.model_path}' (using AutoProcessor)...")
            # Always load processor from the path/ID
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

            if self.is_local:
                # Load config from local directory
                logger.info(f"Loading config from local directory '{self.model_path}'...")
                config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                # Instantiate model architecture from config
                logger.info("Instantiating model from local config...")
                self.model = AutoModel.from_config(config, trust_remote_code=True) 
                # Manually load weights from the specified local file
                weight_file_path = os.path.join(self.model_path, self.weight_filename)
                if not os.path.isfile(weight_file_path):
                    raise FileNotFoundError(f"Weight file not found at: {weight_file_path}")
                logger.info(f"Loading weights manually from '{weight_file_path}'...")
                state_dict = torch.load(weight_file_path, map_location='cpu')
                # Handle potential nested state dict (e.g., if saved as checkpoint['model'])
                if isinstance(state_dict, dict) and any(k in state_dict for k in ['model', 'state_dict', 'module']):
                     potential_keys = ['model', 'state_dict', 'module']
                     for key in potential_keys:
                          if key in state_dict:
                               state_dict = state_dict[key]
                               logger.info(f"Using state_dict from key '{key}' in the loaded file.")
                               break
                # Clean potential 'module.' prefix from DDP saving
                cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Load the state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)
                if missing_keys:
                     logger.warning(f"Missing keys when loading state dict: {missing_keys}")
                if unexpected_keys:
                     logger.warning(f"Unexpected keys when loading state dict: {unexpected_keys}")
                logger.info("Weights loaded manually into instantiated model.")
            else:
                # Load model directly from Hugging Face ID (includes weights)
                logger.info(f"Loading model directly from Hugging Face ID '{self.model_path}'...")
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            logger.info("InternVideo2 model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InternVideo2 model/processor from '{self.model_path}': {e}", exc_info=True)
            raise

        # --- Determine Embedding Dimension and Config---
        self.embedding_dim = None
        try:
             # Use the loaded config (either from local or fetched by AutoModel)
             model_config = self.model.config 
             self.embedding_dim = model_config.hidden_size
        except AttributeError:
             logger.warning("Could not determine embedding dimension from model.config.hidden_size")

        _processor_res = None
        if hasattr(self.processor, 'size') and isinstance(self.processor.size, dict):
            if 'shortest_edge' in self.processor.size:
                 _processor_res = self.processor.size['shortest_edge']
            elif 'height' in self.processor.size and 'width' in self.processor.size:
                  _processor_res = min(self.processor.size['height'], self.processor.size['width']) 

        self.model_config = {
            'resolution': _processor_res or RESOLUTION, # Prioritize processor info
            'num_frames_to_sample': NUM_FRAMES_TO_SAMPLE,
            'batch_size': BATCH_SIZE
            # Normalization is handled by the processor
        }
        logger.info(f"Using effective resolution: {self.model_config['resolution']}")

        if self.embedding_dim:
            logger.info(f"Determined InternVideo2 embedding dimension: {self.embedding_dim}")
        else:
            logger.warning("Could not determine InternVideo2 embedding dimension.")
            # Fallback or raise error

    @torch.no_grad()
    def encode_video(self, video_path):
        """
        Loads, preprocesses a video using the HF processor, extracts features,
        and returns the pooled feature vector.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor or None: The pooled embedding tensor (shape [1, embed_dim]) on CPU,
                                  or None if an error occurs.
        """
        logger.info(f"Encoding video: {video_path} using InternVideo2 ({self.model_path})")
        try:
            # Load and preprocess video frames using the processor
            # Returns tensor, e.g., [1, T, C, H, W] or [B, C, T, H, W]
            pixel_values = _load_and_preprocess_video_internvideo(
                video_path=video_path,
                processor=self.processor,
                num_frames_to_sample=self.model_config['num_frames_to_sample']
            ).to(self.device)

            # --- Feature Extraction ---
            # Pass pixel_values to the model
            # Input format depends on the specific model architecture within transformers
            # Check model documentation if direct `pixel_values=...` doesn't work.
            logger.debug(f"Running model forward pass with pixel_values shape: {pixel_values.shape}")
            outputs = self.model(pixel_values=pixel_values)

            # Extract embeddings - typically last_hidden_state or pooler_output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_embeddings = outputs.pooler_output
                logger.debug("Using pooler_output.")
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                # Average the last hidden state across the sequence dimension (usually dim 1)
                # Shape might be [B, SeqLen, Dim] -> [B, Dim] or [B, T, N, Dim] -> [B, Dim]
                hidden_state = outputs.last_hidden_state
                if hidden_state.ndim == 3: # [B, SeqLen, Dim] -> Average over SeqLen
                    pooled_embeddings = hidden_state.mean(dim=1)
                elif hidden_state.ndim > 3: # Potentially spatio-temporal [B, T, N, Dim] etc.
                     # Average over all sequence/spatial/temporal dimensions except batch and feature dim
                     dims_to_average = tuple(range(1, hidden_state.ndim - 1))
                     pooled_embeddings = hidden_state.mean(dim=dims_to_average)
                else: # Should not happen if ndim >= 2
                     logger.error(f"Unexpected last_hidden_state shape: {hidden_state.shape}")
                     return None
                logger.debug(f"Using mean of last_hidden_state. Original shape: {outputs.last_hidden_state.shape}")
            else:
                logger.error("Model output does not contain 'pooler_output' or 'last_hidden_state'. Cannot extract embeddings.")
                logger.debug(f"Available model output keys: {outputs.keys()}")
                return None

            # Ensure output is [1, D] or [B, D] -> take first item if B > 1 (shouldn't happen with BATCH_SIZE=1)
            if pooled_embeddings.shape[0] > 1:
                 logger.warning(f"Output embedding batch size is {pooled_embeddings.shape[0]} > 1. Taking the first element.")
                 final_embedding = pooled_embeddings[0:1].cpu()
            else:
                 final_embedding = pooled_embeddings.cpu() # Return embedding on CPU
                 
            logger.debug(f"Got final InternVideo2 embedding shape: {final_embedding.shape}")
            return final_embedding

        except FileNotFoundError:
            logger.error(f"Video file not found during encoding: {video_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to encode video {video_path} with InternVideo2: {e}", exc_info=True)
            return None

