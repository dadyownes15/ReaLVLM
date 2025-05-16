import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import KFold
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
from sklearn.pipeline import make_pipeline
# Assuming detector_model.py is in the same package (src.models)
from .detector import Detector 
from .reducer import DimensionReduction

class DetectorTask:
    def __init__(self, detector_model: Detector, training_data_path: str, test_data_path: str, dataset_name: str, encoder_name: str, reducer: DimensionReduction = None):
        self.detector_model = detector_model
        self.training_data_path = Path(training_data_path)
        self.test_data_path = Path(test_data_path)
        self.dataset_name = dataset_name
        self.encoder_name = encoder_name
        
        # Use detector model's actual class name for clarity in paths   
        self.detector_name = self.detector_model.__class__.__name__
        self.task_name = f"{self.dataset_name}_{self.encoder_name}_{self.detector_name}"
        
        self.scores_by_video: Dict[str, torch.Tensor] = {}

        if reducer is not None:
            self.reducer = reducer
        else:
            self.reducer = None

    def _load_all_clip_embeddings(self, data_path: Path) -> List[torch.Tensor]:
        """Loads all .pt files (clips) from all video subdirectories in data_path."""
        all_clip_tensors: List[torch.Tensor] = []
        if not data_path.exists():
            print(f"Data path {data_path} does not exist. Returning empty list.")
            return all_clip_tensors
        if not data_path.is_dir():
            print(f"Data path {data_path} is not a directory. Returning empty list.")
            return all_clip_tensors

        for video_dir in data_path.iterdir():
            if video_dir.is_dir():
                # print(f"Processing video directory: {video_dir.name}")
                for clip_file in video_dir.glob("*.pt"):
                    try:
                        # print(f"Loading clip: {clip_file}")
                        clip_tensor = torch.load(clip_file)
                        if clip_tensor.ndim == 1: # Ensure (1, D_features)
                            clip_tensor = clip_tensor.unsqueeze(0)
                        # Optionally, handle cases like (D_features, 1) if they occur
                        # elif clip_tensor.ndim == 2 and clip_tensor.shape[1] == 1 and clip_tensor.shape[0] > 1:
                        #     clip_tensor = clip_tensor.T 
                        all_clip_tensors.append(clip_tensor)
                    except Exception as e:
                        print(f"Error loading clip {clip_file} in {video_dir.name}: {e}")
        # print(f"Loaded {len(all_clip_tensors)} clip tensors from {data_path}")
        return all_clip_tensors

    def train_detector(self):
        """
        Train the detector on the training data.
        """
        print(f"Starting training for {self.task_name} on dataset {self.dataset_name}...")
        
        if not self.training_data_path.exists() or not any(self.training_data_path.iterdir()):
            print(f"Training data path {self.training_data_path} is empty, does not exist, or contains no subdirectories. Skipping training.")
            return

        all_training_clip_embeddings = self._load_all_clip_embeddings(self.training_data_path)

        # Print shape of the all training clip_embedidngs
        print(f"Training data path {self.training_data_path} contains {len(all_training_clip_embeddings)} clip embeddings.")

        if not all_training_clip_embeddings:
            print(f"No embeddings found in {self.training_data_path} or its subdirectories. Skipping training.")
            return

        try:
            print(f"Training data shape: {all_training_clip_embeddings[0].shape}")
            if self.reducer is not None:
                # Learn reducer
                raise NotImplementedError("Cross-validation and grid search not implemented for reduced dimensions")
                   
                print(f"Fitting reducer...")
                self.reducer.fit(all_training_clip_embeddings)
                # Reduce dimensions
                print(f"Reducing dimensions...")
                all_training_clip_embeddings = self.reducer.reduce_dimensions(all_training_clip_embeddings)
            
        
            self.detector_model.train(all_training_clip_embeddings)
            print(f"Training for {self.task_name} completed.")
        
        except Exception as e:
            print(f"Error during training for {self.task_name}: {e}")


    def test_detector(self):
        """
        Run the detector model on the test data.
        Scores are expected to be per-clip.
        """
        print(f"Starting testing for {self.task_name} on dataset {self.dataset_name}...")
        self.scores_by_video.clear()

        if not self.test_data_path.exists(): # Check existence before iterdir
            print(f"Test data path {self.test_data_path} does not exist. Skipping testing.")
            return

        all_clip_embeddings_list: List[torch.Tensor] = []
        video_metadata: List[Dict[str, Any]] = []

        # Phase 1: Collect all embeddings and their video provenance
        print("Collecting and preparing all test embeddings...")
        
        video_dirs = sorted([d for d in self.test_data_path.iterdir() if d.is_dir()])
        if not video_dirs:
            print(f"No video subdirectories found in {self.test_data_path}. Skipping testing.")
            return

        for video_dir in video_dirs:
            video_stem = video_dir.name
            clips_for_this_video: List[torch.Tensor] = []
            clip_files = sorted(list(video_dir.glob("*.pt")))

            for clip_file in clip_files:
                try:
                    clip_tensor = torch.load(clip_file)
                    
                    if not isinstance(clip_tensor, torch.Tensor):
                        print(f"Warning: File {clip_file} did not load as a torch.Tensor. Got {type(clip_tensor)}. Skipping.")
                        continue
                    
                    if clip_tensor.ndim == 0:
                        print(f"Warning: Clip {clip_file} is a scalar tensor. Skipping.")
                        continue
                    
                    if clip_tensor.ndim == 1:
                        clip_tensor = clip_tensor.unsqueeze(0)
                    
                    for i in range(clip_tensor.shape[0]):
                        single_embedding = clip_tensor[i, :].unsqueeze(0)
                        clips_for_this_video.append(single_embedding)
                        
                except FileNotFoundError:
                    print(f"Error: Clip file {clip_file} not found. Skipping.")
                except Exception as e:
                    print(f"Error loading or processing clip {clip_file} from {video_dir.name}: {e}")
            
            if clips_for_this_video:
                all_clip_embeddings_list.extend(clips_for_this_video)
                video_metadata.append({'name': video_stem, 'num_clips': len(clips_for_this_video)})
            # else: # No print here, could be many such videos
                # print(f"No valid embeddings loaded for video {video_stem} from {video_dir}.")

        if not all_clip_embeddings_list:
            print(f"No embeddings collected from any video in {self.test_data_path.name}. Testing aborted.")
            return 

        print(f"Collected {len(all_clip_embeddings_list)} total clip embeddings from {len(video_metadata)} videos.")

        # Phase 2: Dimension Reduction
        processed_embeddings_list: List[torch.Tensor]
        if self.reducer is not None:
            print("Reducing dimensions of all test embeddings...")
            try:
                processed_embeddings_list = self.reducer.reduce_dimensions(all_clip_embeddings_list)
                print(f"Dimension reduction complete. Output {len(processed_embeddings_list)} embeddings.")
                if len(processed_embeddings_list) != len(all_clip_embeddings_list):
                    print(f"Warning: Number of embeddings changed after reduction: {len(all_clip_embeddings_list)} -> {len(processed_embeddings_list)}. This may cause issues.")
            except Exception as e:
                print(f"Error during dimension reduction for task {self.task_name}: {e}")
                print("Proceeding with original embeddings for score calculation.")
                processed_embeddings_list = all_clip_embeddings_list
        else:
            processed_embeddings_list = all_clip_embeddings_list
            print("No dimension reducer specified. Proceeding with original embeddings.")

        # Phase 3: Score Calculation and Storage
        print("Calculating scores for each video...")
        current_embedding_idx = 0
        for meta in video_metadata:
            video_stem = meta['name']
            num_clips_for_video = meta['num_clips']

            if not (current_embedding_idx + num_clips_for_video <= len(processed_embeddings_list)):
                print(f"Error: Not enough processed embeddings for video {video_stem}. Expected slice up to index {current_embedding_idx + num_clips_for_video -1} but have only {len(processed_embeddings_list)} total. Skipping video.")
                current_embedding_idx += num_clips_for_video # Must advance index regardless
                continue

            video_specific_embeddings_slice = processed_embeddings_list[current_embedding_idx : current_embedding_idx + num_clips_for_video]
            
            if not video_specific_embeddings_slice:
                 print(f"No embeddings available in the processed list for video {video_stem} (slice was empty). Skipping.")
                 current_embedding_idx += num_clips_for_video
                 continue

            try:
                valid_tensors_for_stack = [t for t in video_specific_embeddings_slice if isinstance(t, torch.Tensor) and t.ndim > 0]
                if len(valid_tensors_for_stack) != num_clips_for_video:
                    print(f"Warning: Mismatch in expected ({num_clips_for_video}) and valid ({len(valid_tensors_for_stack)}) tensors for video {video_stem}. Using valid {len(valid_tensors_for_stack)} tensors.")
                    if not valid_tensors_for_stack:
                        print(f"No valid tensors to stack for video {video_stem}. Skipping score calculation.")
                        current_embedding_idx += num_clips_for_video
                        continue
                
                # Assemble the batch for scoring based on whether reduction happened
                if self.reducer is not None and processed_embeddings_list is not all_clip_embeddings_list:
                    # If reducer was successfully applied, embeddings are 1D (D_reduced).
                    # Stack them to make a batch (M, D_reduced).
                    embeddings_for_scoring = torch.stack(valid_tensors_for_stack, dim=0)
                else:
                    # If no reducer or reduction failed, embeddings are (1, D_original).
                    # Concatenate them to make a batch (M, D_original).
                    embeddings_for_scoring = torch.cat(valid_tensors_for_stack, dim=0)

                scores = self.detector_model.calculate_score(embeddings_for_scoring) 
                
                if scores is None:
                    current_embedding_idx += num_clips_for_video
                    continue

                if isinstance(scores, np.ndarray):
                    scores = torch.from_numpy(scores).float()

                if not isinstance(scores, torch.Tensor):
                    print(f"Scores for video {video_stem} are not a torch.Tensor. Got {type(scores)}. Skipping.")
                    current_embedding_idx += num_clips_for_video
                    continue
                
                self.scores_by_video[video_stem] = scores
            except Exception as e:
                print(f"Error calculating scores for video {video_stem}: {e}")
            
            current_embedding_idx += num_clips_for_video
        
        final_processed_videos = len(self.scores_by_video)
        print(f"Testing for {self.task_name} completed. Processed {final_processed_videos} videos with scores out of {len(video_metadata)} videos that had embeddings.")


    def save_output(self):
        """
        Save the scores for each video to:
        output/<dataset_name>/<detector_name>/<video_stem>/scores.pt
        """
        print(f"Saving outputs for {self.task_name}...")
        if not self.scores_by_video:
            print("No scores to save. Run test_detector() first or ensure it produced results.")
            return

        base_output_dir = Path("output") / self.dataset_name / "detector_outputs" / self.detector_name / self.encoder_name
        
        saved_count = 0
        for video_stem, scores_tensor in self.scores_by_video.items():
            video_output_dir = base_output_dir / video_stem
            try:
                video_output_dir.mkdir(parents=True, exist_ok=True)
                save_path = video_output_dir / "scores.pt"
                torch.save(scores_tensor, save_path)
                # print(f"Saved scores for {video_stem} to {save_path}")
                saved_count += 1
            except Exception as e:
                print(f"Error saving scores for video {video_stem} to {video_output_dir/'scores.pt'}: {e}")
        
        print(f"Saving outputs for {self.task_name} complete. Saved scores for {saved_count} videos in {base_output_dir}.")

    def run(self):
        """
        Train the detector.
        Run the detector on the test data.
        Save the output scores.
        """
        print(f"Starting DetectorTask: {self.task_name}")
        self.train_detector()
        self.test_detector()
        self.save_output()
        print(f"DetectorTask {self.task_name} finished.")



cv  = KFold(n_splits=5, shuffle=True, random_state=42)

def avg_margin(estimator, X, _y=None):
    # Larger (less negative) == safer inliers, so we maximize
    return np.mean(estimator.score_samples(X))
