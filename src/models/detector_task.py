## 

"""

DetectorTask(detector_model, training_data, test_data, dataset_name)
    - Gives the task a unique name based on the dataset name and the detector model name, which should be used for saving

train_detector()
    - train the detector on the training data

test_detector()
    
    - Run the detector model on 
    - Output should be(N_Test_Samples, scores_i)
    - scores_i should be a float between 0 and 1, where 0 is normal and 1 is abnormal

save_output()
    - save the output to a file
    - output should be a numpy array of shape (N_Test_Samples, scores_i)

run()
    - train the detector
    - run the detector on the test data
    - save the output


"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

# Assuming detector_model.py is in the same directory or src.models path
try:
    from .detector_model import Detector
except ImportError:
    # Fallback for environments where relative import fails (e.g. running script directly)
    # This assumes src is in PYTHONPATH or script is run from project root.
    from detector_model import Detector 

class DetectorTask:
    def __init__(self, detector_model: Detector, training_data_path: str, test_data_path: str, dataset_name: str):
        self.detector_model = detector_model
        self.training_data_path = Path(training_data_path)
        self.test_data_path = Path(test_data_path)
        self.dataset_name = dataset_name
        
        # Use detector model's actual class name for clarity in paths
        self.detector_name = self.detector_model.__class__.__name__
        self.task_name = f"{self.dataset_name}_{self.detector_name}"
        
        self.scores_by_video: Dict[str, torch.Tensor] = {}

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
                        # Ensure model is on the same device as data if applicable
                        # clip_tensor = torch.load(clip_file, map_location=self.detector_model.device if hasattr(self.detector_model, 'device') else None)
                        clip_tensor = torch.load(clip_file)
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

        if not all_training_clip_embeddings:
            print(f"No embeddings found in {self.training_data_path} or its subdirectories. Skipping training.")
            return

        try:
            # Assuming detector's train method can handle a list of tensors or a single concatenated tensor.
            # If it expects a single tensor, concatenation is needed:
            # Example: concatenated_embeddings = torch.cat(all_training_clip_embeddings, dim=0)
            # For now, passing the list, assuming the model handles it or specific concatenation logic is inside the model.
            # Or, if your model's train method specifically wants concatenated tensors:
            if hasattr(self.detector_model, 'train_on_concatenated') and self.detector_model.train_on_concatenated:
                 concatenated_embeddings = torch.cat(all_training_clip_embeddings, dim=0) # Ensure dim is appropriate
                 print(f"Training with {concatenated_embeddings.shape[0]} concatenated training clips.")
                 self.detector_model.train(concatenated_embeddings)
            else: # Assumes train method can handle list of tensors or individual processing
                 print(f"Training with {len(all_training_clip_embeddings)} training clips (passed as list).")
                 self.detector_model.train(all_training_clip_embeddings)
            print(f"Training for {self.task_name} completed.")
        except Exception as e:
            print(f"Error during training for {self.task_name}: {e}")


    def test_detector(self):
        """
        Run the detector model on the test data, video by video.
        Scores are expected to be per-clip.
        """
        print(f"Starting testing for {self.task_name} on dataset {self.dataset_name}...")
        self.scores_by_video.clear()

        if not self.test_data_path.exists() or not any(self.test_data_path.iterdir()):
            print(f"Test data path {self.test_data_path} is empty, does not exist, or contains no subdirectories. Skipping testing.")
            return

        for video_dir in self.test_data_path.iterdir():
            if not video_dir.is_dir():
                # print(f"Skipping non-directory item: {video_dir.name}")
                continue
            
            video_stem = video_dir.name
            # print(f"Testing video: {video_stem}")
            video_clip_embeddings: List[torch.Tensor] = []
            for clip_file in video_dir.glob("*.pt"):
                try:
                    # clip_tensor = torch.load(clip_file, map_location=self.detector_model.device if hasattr(self.detector_model, 'device') else None)
                    clip_tensor = torch.load(clip_file)
                    video_clip_embeddings.append(clip_tensor)
                except Exception as e:
                    print(f"Error loading clip {clip_file} for video {video_stem}: {e}")
            
            if not video_clip_embeddings:
                print(f"No embeddings found for video {video_stem} in {video_dir}. Skipping this video.")
                continue

            try:
                # Concatenate clips for the current video before passing to calculate_score
                # Assuming calculate_score expects a single tensor for all clips of one video
                video_embeddings_tensor = torch.cat(video_clip_embeddings, dim=0) # Ensure dim is appropriate
                # print(f"Calculating scores for {video_stem} with {video_embeddings_tensor.shape[0]} clips.")
                
                scores = self.detector_model.calculate_score(video_embeddings_tensor) # Expected: (N_clips_in_video, 1) or (N_clips_in_video,)
                
                if scores is None:
                    print(f"Detector returned None for scores for video {video_stem}. Skipping.")
                    continue

                if not isinstance(scores, torch.Tensor):
                    print(f"Scores for video {video_stem} are not a torch.Tensor. Got {type(scores)}. Skipping.")
                    continue
                
                # Basic validation for score shape (optional, but good practice)
                # if scores.ndim > 2 or (scores.ndim == 2 and scores.shape[1] != 1 and scores.shape[0] == video_embeddings_tensor.shape[0]):
                #    print(f"Warning: Unexpected score shape {scores.shape} for video {video_stem} with {video_embeddings_tensor.shape[0]} clips.")

                self.scores_by_video[video_stem] = scores
                # print(f"Stored scores for {video_stem} with shape {scores.shape}")

            except Exception as e:
                print(f"Error calculating scores for video {video_stem}: {e}")
        
        print(f"Testing for {self.task_name} completed. Processed {len(self.scores_by_video)} videos.")


    def save_output(self):
        """
        Save the scores for each video to:
        output/<dataset_name>/<detector_name>/<video_stem>/scores.pt
        """
        print(f"Saving outputs for {self.task_name}...")
        if not self.scores_by_video:
            print("No scores to save. Run test_detector() first or ensure it produced results.")
            return

        base_output_dir = Path("output") / self.dataset_name / self.detector_name
        
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
