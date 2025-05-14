import unittest
import torch
from pathlib import Path
import tempfile
import shutil
from typing import List, Optional, Union
import sys # Add sys import

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Adjust the import path based on your project structure
# If tests directory is at the same level as src:
from src.models.detector_task import DetectorTask
from src.models.detector_model import Detector

class DummyDetector(Detector):
    def __init__(self, name="DummyDetector"):
        super().__init__()
        self.name = name
        self.__class__.__name__ = name # For DetectorTask to get the name
        self.trained_on_embeddings: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        self.scores_calculated_for_video: Optional[torch.Tensor] = None
        self.train_call_count = 0
        self.calculate_score_call_count = 0
        # self.train_on_concatenated = False # Set True if your dummy/real model expects concatenated

    def train(self, embeddings: Union[torch.Tensor, List[torch.Tensor]]):
        self.train_call_count += 1
        self.trained_on_embeddings = embeddings
        # print(f"DummyDetector '{self.name}': train called with embeddings type: {type(embeddings)}")

    def calculate_score(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        self.calculate_score_call_count += 1
        self.scores_calculated_for_video = embeddings
        # print(f"DummyDetector '{self.name}': calculate_score called with embeddings shape: {embeddings.shape}")
        if embeddings.ndim == 0 or embeddings.shape[0] == 0:
            return torch.empty(0, 1) # Return empty tensor for empty input
        # Return scores: (N_clips_in_video, 1)
        return torch.rand(embeddings.shape[0], 1)

    # Implement other abstract methods if any in your Detector base class
    def detect(self, embeddings): # Example
        return self.calculate_score(embeddings)

    def calculate_scores(self, embeddings_dir): # Example from Detector base
        pass


class TestDetectorTask(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.dataset_name = "TestDataset"
        
        self.dummy_detector = DummyDetector(name="TestDetector")

        # Setup paths for dummy data (mimicking EmbeddingTask output)
        self.base_embeddings_path = self.test_dir / "embeddings" / self.dataset_name / "DummyEncoder"
        self.train_embeddings_path = self.base_embeddings_path / "training"
        self.test_embeddings_path = self.base_embeddings_path / "test"

        self.train_embeddings_path.mkdir(parents=True, exist_ok=True)
        self.test_embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Expected output path for scores
        self.expected_scores_base_path = Path("output") / self.dataset_name / self.dummy_detector.name

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if self.expected_scores_base_path.exists():
            shutil.rmtree(Path("output") / self.dataset_name) # Clean up dataset output

    def _create_dummy_video_embeddings(self, video_base_path: Path, video_name: str, num_clips: int, embed_dim: int = 128):
        video_dir = video_base_path / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_clips):
            # Create a tensor with shape (num_features_per_clip, embed_dim)
            # For DetectorTask, it expects each .pt file to be a single clip's embedding,
            # often (1, embed_dim) or (embed_dim,). We'll use (1, embed_dim) which is then concatenated.
            dummy_clip_tensor = torch.randn(1, embed_dim) 
            torch.save(dummy_clip_tensor, video_dir / f"clip_{i}.pt")
        return video_dir

    def test_initialization(self):
        task = DetectorTask(
            detector_model=self.dummy_detector,
            training_data_path=str(self.train_embeddings_path),
            test_data_path=str(self.test_embeddings_path),
            dataset_name=self.dataset_name
        )
        self.assertEqual(task.dataset_name, self.dataset_name)
        self.assertEqual(task.detector_name, self.dummy_detector.name)
        self.assertEqual(task.task_name, f"{self.dataset_name}_{self.dummy_detector.name}")
        self.assertIsInstance(task.training_data_path, Path)
        self.assertIsInstance(task.test_data_path, Path)

    def test_load_all_clip_embeddings(self):
        task = DetectorTask(self.dummy_detector, str(self.train_embeddings_path), str(self.test_embeddings_path), self.dataset_name)
        
        # Test with empty directory
        loaded_clips = task._load_all_clip_embeddings(self.train_embeddings_path)
        self.assertEqual(len(loaded_clips), 0)

        # Test with non-existent path
        non_existent_path = self.test_dir / "non_existent"
        loaded_clips_non_existent = task._load_all_clip_embeddings(non_existent_path)
        self.assertEqual(len(loaded_clips_non_existent), 0)

        # Test with data
        self._create_dummy_video_embeddings(self.train_embeddings_path, "video1", 3)
        self._create_dummy_video_embeddings(self.train_embeddings_path, "video2", 2)
        loaded_clips_with_data = task._load_all_clip_embeddings(self.train_embeddings_path)
        self.assertEqual(len(loaded_clips_with_data), 5) # 3 + 2
        for clip in loaded_clips_with_data:
            self.assertIsInstance(clip, torch.Tensor)
            self.assertEqual(clip.shape, (1, 128)) # As created by _create_dummy_video_embeddings

    def test_train_detector(self):
        # No training data
        task = DetectorTask(self.dummy_detector, str(self.train_embeddings_path), str(self.test_embeddings_path), self.dataset_name)
        task.train_detector()
        self.assertEqual(self.dummy_detector.train_call_count, 0) # Should skip if no data

        # With training data
        self._create_dummy_video_embeddings(self.train_embeddings_path, "train_video1", 3)
        self.dummy_detector.train_call_count = 0 # Reset
        task.train_detector()
        self.assertEqual(self.dummy_detector.train_call_count, 1)
        self.assertIsNotNone(self.dummy_detector.trained_on_embeddings)
        if isinstance(self.dummy_detector.trained_on_embeddings, list): # Default behavior in DetectorTask
             self.assertEqual(len(self.dummy_detector.trained_on_embeddings), 3)
        elif isinstance(self.dummy_detector.trained_on_embeddings, torch.Tensor): # If train_on_concatenated=True
             self.assertEqual(self.dummy_detector.trained_on_embeddings.shape[0], 3)


    def test_test_detector(self):
        # No test data
        task = DetectorTask(self.dummy_detector, str(self.train_embeddings_path), str(self.test_embeddings_path), self.dataset_name)
        task.test_detector()
        self.assertEqual(self.dummy_detector.calculate_score_call_count, 0)
        self.assertEqual(len(task.scores_by_video), 0)

        # With test data
        self._create_dummy_video_embeddings(self.test_embeddings_path, "test_videoA", 4)
        self._create_dummy_video_embeddings(self.test_embeddings_path, "test_videoB", 0) # Video with no clips
        self._create_dummy_video_embeddings(self.test_embeddings_path, "test_videoC", 2)
        
        self.dummy_detector.calculate_score_call_count = 0 # Reset
        task.test_detector()
        
        self.assertEqual(self.dummy_detector.calculate_score_call_count, 2) # videoB has no clips, so not called
        self.assertIn("test_videoA", task.scores_by_video)
        self.assertIn("test_videoC", task.scores_by_video)
        self.assertNotIn("test_videoB", task.scores_by_video) # Should not be present if no clips
        
        self.assertIsInstance(task.scores_by_video["test_videoA"], torch.Tensor)
        self.assertEqual(task.scores_by_video["test_videoA"].shape, (4, 1)) # N_clips, 1
        self.assertEqual(task.scores_by_video["test_videoC"].shape, (2, 1))

    def test_save_output(self):
        task = DetectorTask(self.dummy_detector, str(self.train_embeddings_path), str(self.test_embeddings_path), self.dataset_name)
        
        # No scores to save
        task.save_output()
        self.assertFalse(self.expected_scores_base_path.exists())

        # With scores
        task.scores_by_video = {
            "videoX": torch.rand(5, 1),
            "videoY": torch.rand(3, 1)
        }
        task.save_output()
        
        self.assertTrue(self.expected_scores_base_path.exists())
        videoX_output_dir = self.expected_scores_base_path / "videoX"
        videoY_output_dir = self.expected_scores_base_path / "videoY"
        self.assertTrue(videoX_output_dir.exists())
        self.assertTrue((videoX_output_dir / "scores.pt").exists())
        self.assertTrue(videoY_output_dir.exists())
        self.assertTrue((videoY_output_dir / "scores.pt").exists())

        loaded_scores_X = torch.load(videoX_output_dir / "scores.pt")
        self.assertTrue(torch.equal(loaded_scores_X, task.scores_by_video["videoX"]))

    def test_run_full_cycle(self):
        # Populate with some training and test data
        self._create_dummy_video_embeddings(self.train_embeddings_path, "train_vid1", 2)
        self._create_dummy_video_embeddings(self.test_embeddings_path, "test_vidA", 3)
        self._create_dummy_video_embeddings(self.test_embeddings_path, "test_vidB", 1)

        task = DetectorTask(
            detector_model=self.dummy_detector,
            training_data_path=str(self.train_embeddings_path),
            test_data_path=str(self.test_embeddings_path),
            dataset_name=self.dataset_name
        )
        task.run()

        # Check if detector methods were called
        self.assertEqual(self.dummy_detector.train_call_count, 1)
        self.assertEqual(self.dummy_detector.calculate_score_call_count, 2) # For test_vidA and test_vidB

        # Check if scores were stored internally (optional, as save_output is the main check here)
        self.assertIn("test_vidA", task.scores_by_video)
        self.assertIn("test_vidB", task.scores_by_video)

        # Check if output files were created
        self.assertTrue((self.expected_scores_base_path / "test_vidA" / "scores.pt").exists())
        self.assertTrue((self.expected_scores_base_path / "test_vidB" / "scores.pt").exists())
        
        loaded_scores_A = torch.load(self.expected_scores_base_path / "test_vidA" / "scores.pt")
        self.assertEqual(loaded_scores_A.shape, (3,1))

if __name__ == '__main__':
    unittest.main() 