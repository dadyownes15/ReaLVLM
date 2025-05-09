import torch
import logging
import os
from pathlib import Path
import sys
import numpy as np
from torchvision import transforms
from decord import VideoReader, cpu
import logging

logger = logging.getLogger(__name__)

# Attempt to import the FluxViT model definition
# This assumes 'fluxvit_model_def.py' is at the project root and project root is in sys.path
try:
    import fluxvit_model_def # Direct import if in sys.path
except ImportError as e:
    logger.error("FluxViT model definition (fluxvit_model_def.py) not found or not importable. "
                 "Ensure it's at the project root and the project root is in sys.path. Error: %s", e)
    raise ImportError("FluxViT model definition (fluxvit_model_def.py) not found. "
                      "Please ensure it's at the project root and the project root is in sys.path.") from e

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Encoder Class ---
class FluxViTEncoder:
    """Encapsulates FluxViT model loading and video embedding extraction."""

    def __init__(self, config_dict):
        """
        Initializes the FluxViTEncoder using a configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing all necessary parameters.
                                Expected keys are defined in a JSON configuration file.
        """
        logger.info("Initializing FluxViTEncoder from configuration dictionary...")
        params = config_dict.get("parameters")
        if params is None:
            raise ValueError("Config missing 'parameters' key.")
        preproc_params = params.get("preprocessing")
        if preproc_params is None:
            raise ValueError("Config missing 'parameters.preprocessing' key.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"FluxViTEncoder determined to use device: {self.device}")

        # --- Extract and Store Configuration ---
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust if needed

        model_param_keys = [
            'model_name',  # e.g., 'fluxvit_small_patch14'
            'checkpoint_path', # Path to the .pth file
            'checkpoint_key',  # Key in the checkpoint dict, e.g., 'model', 'state_dict'
            'model_frames_per_clip', # Max frames the model was trained with for a clip
            'tubelet_size', # From model config
            'patch_size', # From model config
            'img_size', # From model config (spatial resolution)
            'model_kwargs', # Additional kwargs for model constructor
        ]
        preproc_keys = [
            'resolution', # Target resolution for preprocessing (spatial)
            'clip_duration_frames', # How many frames to extract for each clip from video
            'clip_stride', # Stride between start of clips
            'sampling_method', # e.g., 'stride', 'linspace'
            'resize_method', # e.g., 'shortest_edge_then_crop'
            'norm_mean',
            'norm_std'
        ]
        missing_model_keys = [key for key in model_param_keys if key not in params]
        missing_preproc_keys = [key for key in preproc_keys if key not in preproc_params]

        if missing_model_keys:
            raise ValueError(f"Missing required model config keys: {missing_model_keys}")
        if missing_preproc_keys:
             raise ValueError(f"Missing required preprocessing config keys: {missing_preproc_keys}")

        # Store parameters (resolving paths)
        _ckpt_path_rel = params['checkpoint_path']
        self.checkpoint_path = os.path.join(_project_root, _ckpt_path_rel) if not os.path.isabs(_ckpt_path_rel) else _ckpt_path_rel

        self.model_name = params['model_name']
        self.model_frames_per_clip = params['model_frames_per_clip'] # Model's internal frame capacity
        self.tubelet_size = params['tubelet_size']
        self.patch_size = params['patch_size']
        self.img_size = params['img_size'] # Should match preproc_params['resolution'] after crop
        self.checkpoint_key = params.get('checkpoint_key', 'model') # Default to 'model' or 'state_dict'
        self.model_kwargs = params.get('model_kwargs', {})
        
        self.preproc_params = preproc_params
        
        if not os.path.isfile(self.checkpoint_path):
            logger.warning(f"FluxViT checkpoint path not found: {self.checkpoint_path}")
            # raise FileNotFoundError(f"FluxViT checkpoint path not found: {self.checkpoint_path}")


        # --- Initialize Encoder ---
        self.model = self._init_fluxvit_model()
        self.model.eval()

        # The embedding dim for FluxViT is typically `clip_embed_dim` from its config
        # We'll try to infer it or require it in config. For now, assuming it's in model_kwargs or known.
        # FluxViT's 'clip_projector' output dimension is 'clip_embed_dim'
        # Defaulting to a common value if not found, but should be accurate.
        self.embedding_dim = self.model_kwargs.get('clip_embed_dim', 768) # Example, should come from model actual config
        if hasattr(self.model, 'clip_embed_dim'): # If model stores it
             self.embedding_dim = self.model.clip_embed_dim
        elif hasattr(self.model, 'embed_dim') and not hasattr(self.model, 'clip_projector'):
             # If no clip_projector, then embed_dim is the output (unlikely for published models)
             self.embedding_dim = self.model.embed_dim
        else:
            # Attempt to get it from a dummy forward or inspecting clip_projector
            try:
                dummy_output_module = self.model.head[0] # The nn.Linear after projection
                if hasattr(dummy_output_module, 'in_features'):
                    self.embedding_dim = dummy_output_module.in_features
                else: # Fallback, user should verify or set explicitly
                    logger.warning(f"Could not definitively infer embedding_dim. Using default {self.embedding_dim}. Please verify.")
            except Exception:
                 logger.warning(f"Could not definitively infer embedding_dim. Using default {self.embedding_dim}. Please verify.")


        logger.info(f"Initialized FluxViTEncoder. Output embedding dimension: {self.embedding_dim}")


    def _init_fluxvit_model(self):
        """Initializes the FluxViT model and loads pretrained weights."""
        logger.info(f"Initializing FluxViT model: {self.model_name}")
        if not hasattr(fluxvit_model_def, self.model_name):
            raise ValueError(f"Model name '{self.model_name}' not found in fluxvit_model_def.py")

        # These are passed to the FluxViT constructor.
        # They should align with the chosen pre-trained model's architecture.
        # The `img_size` and `num_frames` here are more about model architecture definition
        # than strict input size for PatchEmbed, which can be somewhat flexible or handled by pos_embed interpolation.
        # However, it's best to match them to the pre-training config if known.
        model_constructor_kwargs = {
            'img_size': self.img_size, # Spatial dimensions
            'patch_size': self.patch_size,
            'num_frames': self.model_frames_per_clip, # Temporal dimension for model
            'tubelet_size': self.tubelet_size,
            **(self.model_kwargs or {})
        }
        
        # Remove num_classes if we are not using the head for classification
        if 'num_classes' not in model_constructor_kwargs:
            model_constructor_kwargs['num_classes'] = 0 # Avoid creating a large head if not needed


        model = getattr(fluxvit_model_def, self.model_name)(**model_constructor_kwargs)
        
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            state_dict = None
            if isinstance(checkpoint, dict):
                if self.checkpoint_key in checkpoint:
                    state_dict = checkpoint[self.checkpoint_key]
                elif 'state_dict' in checkpoint: # Common alternative
                    state_dict = checkpoint['state_dict']
                else:
                    # Try to see if any key looks like a state_dict (e.g. contains 'patch_embed.proj.weight')
                    for key, value in checkpoint.items():
                        if isinstance(value, dict) and 'patch_embed.proj.weight' in value:
                            state_dict = value
                            logger.info(f"Found state_dict under key '{key}' by inspecting content.")
                            break
                    if state_dict is None: # If still not found, assume checkpoint IS the state_dict
                        logger.info(f"Checkpoint key '{self.checkpoint_key}' or 'state_dict' not found. Assuming checkpoint is the state_dict itself.")
                        state_dict = checkpoint
            else: # If checkpoint is not a dict, assume it's the state_dict
                state_dict = checkpoint
            
            if state_dict:
                # Clean prefixes (e.g., 'module.' from DDP, or other prefixes like 'encoder.')
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k
                    if new_k.startswith('module.'):
                        new_k = new_k[len('module.'):]
                    if new_k.startswith('encoder.'): # Common in some pre-training frameworks
                        new_k = new_k[len('encoder.'):]
                    # FluxViT specific: sometimes weights are under 'model.'
                    if new_k.startswith('model.'):
                         new_k = new_k[len('model.'):]
                    cleaned_state_dict[new_k] = v
                
                # Adjust for num_classes=0 if original checkpoint had a head
                # This is a common adjustment when using a model as a feature extractor
                current_model_dict = model.state_dict()
                keys_to_remove_from_checkpoint = []
                for k in cleaned_state_dict.keys():
                    if k.startswith('head.'): # if original checkpoint has head layers
                        if k not in current_model_dict: # and current model (num_classes=0) doesn't
                            keys_to_remove_from_checkpoint.append(k)
                            logger.info(f"Removing head key from checkpoint: {k} as current model has no/different head.")
                for k in keys_to_remove_from_checkpoint:
                    del cleaned_state_dict[k]

                msg = model.load_state_dict(cleaned_state_dict, strict=False)
                logger.info(f"Loaded checkpoint with message: {msg}")
                if msg.missing_keys:
                    logger.warning(f"Missing keys in checkpoint: {msg.missing_keys}")
                if msg.unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {msg.unexpected_keys}")
            else:
                logger.error("Could not extract state_dict from checkpoint.")
                raise ValueError("Failed to load state_dict from checkpoint.")
        else:
            logger.warning("No checkpoint path provided or file not found. Model initialized with random weights.")

        model.to(self.device)
        model.eval()
        return model

    def _preprocess_clip(self, clip_frames):
        """
        (Private) Preprocesses a batch/list of frames for a single clip.
        Placeholder: This should be adapted from FluxViT's official preprocessing.
        """
        logger.warning("Using placeholder preprocessing for FluxViT. "
                       "Replace with official preprocessing logic from the FluxViT repository "
                       "for optimal performance with pre-trained models.")

        # --- ATTENTION: Replace this with FluxViT's actual preprocessing ---
        # The following is a generic example inspired by JepaEncoder and common ViT practice.
        # It LIKELY DOES NOT MATCH FluxViT's exact requirements.

        target_resolution = self.preproc_params['resolution'] # Spatial resolution after crop
        num_frames_in_clip = self.preproc_params['clip_duration_frames'] # Frames sampled from video for one clip
        norm_mean = self.preproc_params['norm_mean']
        norm_std = self.preproc_params['norm_std']
        
        # Assuming clip_frames is a NumPy array: (T, H, W, C) from Decord
        frames_tensor = torch.from_numpy(clip_frames).permute(0, 3, 1, 2).float() / 255.0
        # Current shape: T, C, H, W
        
        # 1. Frame Sampling (if clip_frames contains more/less than model_frames_per_clip)
        #    FluxViT's 'num_frames' in its config usually refers to the number of frames *after* considering tubelet_size.
        #    The `clip_duration_frames` from preprocessing config should be what we sample from the video.
        #    This sampled clip is then fed to the model. The model internally processes it based on its `num_frames` and `tubelet_size`.
        #    Ensure `num_frames_in_clip` is appropriate for `self.model_frames_per_clip` (model's capacity)
        #    and `self.tubelet_size`. E.g. model expects T_model = num_frames_in_clip / tubelet_size input tokens temporally.
        
        current_num_frames = frames_tensor.shape[0]
        if current_num_frames != num_frames_in_clip:
            logger.warning(f"Input clip has {current_num_frames} frames, but preproc expected {num_frames_in_clip}. "
                           "Ensure frame sampling in video reading is correct. Truncating/padding if necessary for demo.")
            if current_num_frames > num_frames_in_clip:
                frames_tensor = frames_tensor[:num_frames_in_clip]
            else: # Pad if too short (e.g. with last frame) - simple padding
                padding = torch.zeros(num_frames_in_clip - current_num_frames, *frames_tensor.shape[1:])
                if current_num_frames > 0:
                    padding = frames_tensor[-1:].repeat(num_frames_in_clip - current_num_frames, 1, 1, 1)
                frames_tensor = torch.cat((frames_tensor, padding), dim=0)


        # 2. Spatial Transforms (Resize, Crop)
        #    Common practice: Resize shortest edge, then center crop.
        #    FluxViT's `img_size` parameter is the expected spatial input size.
        transform_list = [
            transforms.Resize(target_resolution, antialias=True), # Resize shortest side to target_resolution
            transforms.CenterCrop(target_resolution),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ]
        transform = transforms.Compose(transform_list)
        
        # Apply transform to each frame
        frames_processed = torch.stack([transform(frame) for frame in frames_tensor]) # T, C, H, W
        
        # Format for model: B, C, T, H, W
        frames_final = frames_processed.permute(1, 0, 2, 3) # C, T, H, W
        return frames_final.unsqueeze(0) # Add Batch dim -> [1, C, T, H, W]


    @torch.no_grad()
    def encode_video(self, video_path):
        """
        Loads, preprocesses (clip-by-clip), and extracts embeddings for a video.
        The embedding is the output of FluxViT's `clip_projector`.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[torch.Tensor] or None: A list of PyTorch Tensors, where each tensor
                                        is an embedding for a clip (shape [embed_dim]) on CPU,
                                        or None if an error occurs or no clips are generated.
        """
        logger.info(f"Encoding video with FluxViT (clip-based): {video_path}")
        try:
            vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
            video_len = len(vr)
            if video_len == 0:
                logger.warning(f"Video has 0 frames: {video_path}")
                return None
            
            clip_len_frames = self.preproc_params['clip_duration_frames'] # Frames to sample per clip
            stride = self.preproc_params['clip_stride']
            
            clip_embeddings = []
            clip_start_indices = list(range(0, video_len - clip_len_frames + 1, stride))
            if not clip_start_indices and video_len >= clip_len_frames : # Ensure last possible clip if stride misses
                 if (video_len - clip_len_frames) % stride != 0:
                      clip_start_indices.append(video_len - clip_len_frames)
                      clip_start_indices = sorted(list(set(clip_start_indices)))

            if not clip_start_indices: # Handle very short videos (shorter than one clip_len_frames)
                 logger.warning(f"Video length ({video_len}) shorter than clip sample length ({clip_len_frames}). "
                                f"Processing available {video_len} frames as a single clip.")
                 frame_indices = np.arange(video_len)
                 frames = vr.get_batch(frame_indices).asnumpy()
                 if frames.shape[0] > 0:
                      # Pad or adjust frames to match `clip_len_frames` if _preprocess_clip expects fixed size
                      # For now, _preprocess_clip has basic truncation/padding.
                      input_tensor = self._preprocess_clip(frames).to(self.device)
                      
                      # Use return_cls=True, return_projected=True to get the pooled output
                      # The second element of the tuple will be 'x_final' from FluxViT's forward
                      _, pooled_features = self.model(input_tensor, return_cls=True, return_projected=True)
                      clip_embeddings.append(pooled_features.squeeze(0).cpu())
                 else:
                      logger.error(f"Could not read any frames from short video: {video_path}")
                      return None
            else:
                logger.info(f"Processing {len(clip_start_indices)} clips for video: {video_path}")
                for start_idx in clip_start_indices:
                    frame_indices = np.arange(start_idx, start_idx + clip_len_frames)
                    frames = vr.get_batch(frame_indices).asnumpy() # T, H, W, C
                    
                    input_tensor = self._preprocess_clip(frames).to(self.device) # 1, C, T, H, W
                    
                    # Verify consistency: preprocessed clip frame count should be what model was configured for
                    # This check is more about the `self.model_frames_per_clip` which is a model architecture param.
                    # The `input_tensor.shape[2]` IS `clip_len_frames` due to preprocessing.
                    # The model's internal `num_frames` param (passed to its constructor) should be compatible.
                    if input_tensor.shape[2] != self.preproc_params['clip_duration_frames']:
                         logger.error(f"Preprocessed clip tensor frame count ({input_tensor.shape[2]}) "
                                      f"doesn't match config clip_duration_frames ({self.preproc_params['clip_duration_frames']}). Skipping clip.")
                         continue
                         
                    # Get pooled features from FluxViT
                    # The forward pass parameters `num_merging_to`, `masking_ratio` are important for FluxViT.
                    # For a generic encoder, we likely use defaults (None) unless specific behavior is desired.
                    # `output_head` default is 0. `align_proj` default is 0.
                    _, pooled_features = self.model(
                        input_tensor, 
                        num_merging_to=None, # Or configure if needed
                        masking_ratio=None,  # Or configure if needed
                        output_head=0,
                        return_cls=True, 
                        return_projected=True,
                        align_proj=0 
                    ) # pooled_features is x_final, shape (B, clip_embed_dim)

                    clip_embeddings.append(pooled_features.squeeze(0).cpu())
                    logger.debug(f"Stored embedding for clip starting at frame {start_idx}.")

            if not clip_embeddings:
                logger.warning(f"No embeddings generated for video: {video_path}")
                return None

            logger.info(f"Finished encoding video. Generated {len(clip_embeddings)} clip embeddings.")
            return clip_embeddings

        except FileNotFoundError:
            logger.error(f"Video file not found during encoding: {video_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to encode video {video_path}: {e}", exc_info=True)
            return None

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # This is a mock example.
    # You would load your actual JSON configuration.
    mock_config = {
        "parameters": {
            "model_name": "fluxvit_small_patch14", # Must match a function in fluxvit_model_def.py
            "checkpoint_path": "path/to/your/fluxvit_small_patch14_checkpoint.pth", # IMPORTANT: Replace this
            "checkpoint_key": "model", # Or 'state_dict', or whatever key holds the weights in the .pth
            "model_frames_per_clip": 16, # Example: Max frames model's PatchEmbed expects (e.g., num_frames for model constructor)
            "tubelet_size": 1,           # Example: From FluxViT model config
            "patch_size": 14,            # Example: From FluxViT model config
            "img_size": 224,             # Example: Spatial res for model constructor (e.g., 224 for 224x224)
            "model_kwargs": {            # Kwargs for the FluxViT model constructor from fluxvit_model_def
                "clip_embed_dim": 768,   # Example: Output dimension of clip_projector
                "depth": 12,             # Example: For fluxvit_small_patch14
                "num_heads": 6,          # Example: For fluxvit_small_patch14
                # Add other necessary kwargs for the specific FluxViT variant being loaded
            }
        },
        "preprocessing": {
            "resolution": 224,             # Preprocessing target spatial resolution
            "clip_duration_frames": 16,    # Frames to extract per clip from video
            "clip_stride": 8,              # Stride between clips
            "sampling_method": "stride",   # (Currently basic stride in encode_video)
            "resize_method": "shortest_edge_then_crop", # (Currently simple resize then crop in _preprocess_clip)
            "norm_mean": [0.485, 0.456, 0.406], # Example ImageNet mean
            "norm_std": [0.229, 0.224, 0.225]   # Example ImageNet std
        }
    }

    # Create a dummy checkpoint file for the example to run without error if path doesn't exist
    # In a real scenario, params['checkpoint_path'] should point to a valid checkpoint.
    dummy_ckpt_path = mock_config["parameters"]["checkpoint_path"]
    if not os.path.exists(dummy_ckpt_path):
        logger.warning(f"Dummy checkpoint path '{dummy_ckpt_path}' does not exist. Creating a placeholder.")
        # Try to create the directory if it doesn't exist
        os.makedirs(os.path.dirname(dummy_ckpt_path), exist_ok=True)
        # Create a minimal state_dict that might load with strict=False if model is small
        # For a real model, this dummy checkpoint is insufficient.
        try:
            # Instantiate a temporary model to get its state_dict structure
            # This part is tricky for a truly generic dummy and might fail
            # if model_kwargs are not perfectly set for a parameterless instantiation.
            # temp_model_kwargs = {**mock_config["parameters"]["model_kwargs"]}
            # temp_model_kwargs.pop('clip_embed_dim', None) # clip_embed_dim might be inferred or part of a sub-module
            # temp_model = getattr(fluxvit_model_def, mock_config["parameters"]["model_name"])(
            #     img_size=mock_config["parameters"]["img_size"],
            #     patch_size=mock_config["parameters"]["patch_size"],
            #     num_frames=mock_config["parameters"]["model_frames_per_clip"],
            #     tubelet_size=mock_config["parameters"]["tubelet_size"],
            #     num_classes=0, # No head for dummy
            #     **temp_model_kwargs
            # )
            # torch.save({mock_config["parameters"]["checkpoint_key"]: temp_model.state_dict()}, dummy_ckpt_path)
            # logger.info(f"Created a placeholder dummy checkpoint at: {dummy_ckpt_path}")
            # Simplified dummy checkpoint:
            torch.save({mock_config["parameters"]["checkpoint_key"]: {}}, dummy_ckpt_path) # Empty state dict
            logger.info(f"Created a placeholder empty dummy checkpoint at: {dummy_ckpt_path}. "
                        "Model will load with random weights unless a real checkpoint is provided.")
        except Exception as e:
            logger.error(f"Could not create dummy model or checkpoint: {e}. The example might fail if checkpoint is required.")


    # Create a dummy video file
    dummy_video_path = "dummy_video.mp4"
    if not os.path.exists(dummy_video_path):
        try:
            # This requires FFMPEG to be installed and accessible.
            # Creates a 1-second black video at 10 FPS, 32x32 resolution.
            # For more robust dummy video creation, consider specific libraries or ensure ffmpeg is in PATH.
            os.system(f"ffmpeg -y -f lavfi -i color=c=black:s=32x32:r=10 -t 1 {dummy_video_path} -hide_banner -loglevel error")
            if os.path.exists(dummy_video_path):
                 logger.info(f"Created dummy video: {dummy_video_path}")
            else:
                 logger.warning(f"Failed to create dummy video '{dummy_video_path}'. "
                                 "The example might fail if a video file is required.")
        except Exception as e:
            logger.warning(f"Could not create dummy video using ffmpeg: {e}. The example might fail.")

    if os.path.exists(dummy_ckpt_path) and os.path.exists(dummy_video_path):
        try:
            encoder = FluxViTEncoder(mock_config)
            logger.info(f"FluxViTEncoder initialized with embedding dimension: {encoder.embedding_dim}")
            
            embeddings = encoder.encode_video(dummy_video_path)
            if embeddings:
                logger.info(f"Successfully encoded dummy video. Number of clip embeddings: {len(embeddings)}")
                logger.info(f"Shape of first embedding: {embeddings[0].shape}")
            else:
                logger.warning("Encoding dummy video returned no embeddings.")
        except Exception as e:
            logger.error(f"Error during example FluxViTEncoder usage: {e}", exc_info=True)
    else:
        logger.warning("Dummy checkpoint or dummy video not found, skipping example usage.")

    # Clean up dummy files
    # if os.path.exists(dummy_video_path):
    #     os.remove(dummy_video_path)
    # if mock_config["parameters"]["checkpoint_path"] == dummy_ckpt_path and os.path.exists(dummy_ckpt_path) and not "path/to/your" in dummy_ckpt_path:
    #     # Only remove if it's the one we created and not a user's placeholder path
    #     os.remove(dummy_ckpt_path) 