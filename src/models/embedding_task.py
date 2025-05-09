import json
import os
from pathlib import Path
import torch

from src.models.encoder_model import Encoder


class EmbeddingTask:
    def __init__(self, model_config_path: str, dataset_name: str, training_data_path: str , test_data_path: str):
        self.model_config_path = model_config_path
        self.dataset_name = dataset_name
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.model_config = self._load_model_config(model_config_path)
        self.encoder_name = self.model_config.get('model_name', 'default_encoder') # Assuming model_name is in config
        self.output_path = Path('output') / self.dataset_name / self.encoder_name
        self.checkpoint_path = self.output_path / 'checkpoint.json'
        self.checkpoint_data = self._check_embedding_checkpoint()

    def run(self):
        print(f"Starting embedding task for {self.dataset_name} with {self.encoder_name} using {self.model_config}")
        
        encoder = self._load_encoder(self.model_config)

        dataset_splits = self._load_dataset()

        for split, item_paths_list in dataset_splits.items():
            if item_paths_list is None: 
                print(f"Skipping {split} split as no data paths were found or path was invalid.")
                continue
            
            # Ensure structures for the current split exist
            if split not in self.checkpoint_data['embeddings']:
                self.checkpoint_data['embeddings'][split] = {}
            if split not in self.checkpoint_data['failed_items']:
                self.checkpoint_data['failed_items'][split] = {}

            for item_path in item_paths_list:
                item_stem = item_path.stem

                if self.checkpoint_data['embeddings'][split].get(item_stem) == 'processed':
                    print(f"Skipping already processed: {split}/{item_path.name}")
                    continue
                
                # If previously failed, it will be retried. 
                # If successful now, it will be removed from failed_items later.
                
                print(f"Processing: {split}/{item_path.name} (stem: {item_stem})")
                try:
                    encoded_results = encoder.encode(item_path)
                    
                    if encoded_results is None:
                        print(f"Skipping {split}/{item_path.name} – encode() failed.")
                        self.checkpoint_data['failed_items'][split][item_stem] = "encode() returned None"
                        continue

                    if isinstance(encoded_results, list):
                        if not encoded_results: 
                            print(f"Warning: encoder.encode returned an empty list for {item_path.name}. Skipping.")
                            continue
                        for i, single_embedding in enumerate(encoded_results):
                            self._save_encoding(single_embedding, split, item_path, i)
                    else:
                        self._save_encoding(encoded_results, split, item_path, 0) 

                    # Mark as processed
                    self.checkpoint_data['embeddings'][split][item_stem] = 'processed'
                    # If it was previously in failed_items, remove it
                    if item_stem in self.checkpoint_data['failed_items'][split]:
                        del self.checkpoint_data['failed_items'][split][item_stem]
                        print(f"Removed {item_stem} from failed_items list after successful processing.")
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Error processing or saving {split}/{item_path.name}: {error_message}")
                    self.checkpoint_data['failed_items'][split][item_stem] = error_message
                    # Ensure it's not marked as processed if it failed
                    if item_stem in self.checkpoint_data['embeddings'][split]:
                        del self.checkpoint_data['embeddings'][split][item_stem]
                finally:
                    # Save checkpoint after each item attempt (success or failure)
                    with open(self.checkpoint_path, 'w') as f:
                        json.dump(self.checkpoint_data, f, indent=4)
                    # print(f"Checkpoint updated for {item_path.name}") # Removed as per user edit

        print("Embedding task finished.")

    def _load_encoder(self, model_config: dict):
        print(f"Loading model with config: {model_config}")
        encoder = Encoder(**model_config)
        return encoder
    
    def _load_dataset(self):
        print(f"Loading dataset structure from training: {self.training_data_path} and test: {self.test_data_path}")
        dataset_splits = {
            'test': [], # Now a list of paths
            'training': [] # Now a list of paths
        }

        # Helper function to populate splits
        def populate_split(split_name, data_dir):
            data_path = Path(data_dir)
            if not data_path.exists() or not data_path.is_dir():
                print(f"Warning: {split_name} data path {data_path} does not exist or is not a directory. Skipping.")
                dataset_splits[split_name] = None # Mark as None if path is invalid
                return
            
            split_files = []
            for item in data_path.iterdir():
                if item.is_file(): # Assuming each file is a video to be processed
                    # Optionally, add filtering by extension, e.g.:
                    # if item.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    split_files.append(item)
                else:
                    print(f"Warning: Skipping non-file item {item} in {data_path}")
            dataset_splits[split_name] = split_files
        
        populate_split('test', self.test_data_path)
        populate_split('training', self.training_data_path)

        if (dataset_splits['test'] is None or not dataset_splits['test']) and \
           (dataset_splits['training'] is None or not dataset_splits['training']):
            print("Warning: No data files found in the provided training or test paths.")
        else:
            print("Dataset structure loaded.")
            if dataset_splits['test'] is not None:
                 print(f"Found {len(dataset_splits['test'])} files in test set.")
            if dataset_splits['training'] is not None:
                 print(f"Found {len(dataset_splits['training'])} files in training set.")
        return dataset_splits

    def _save_encoding(self, embedding_data, split: str, original_item_path: Path, clip_index: int):
        # Saves the embedding to: output/<dataset_name>/<encoder_name>/<split>/<original_item_stem>/clip_<clip_index>.pt
        video_specific_dir = self.output_path / split / original_item_path.stem
        video_specific_dir.mkdir(parents=True, exist_ok=True)
        
        embedding_file_name = f"clip_{clip_index}.pt"
        embedding_file_path = video_specific_dir / embedding_file_name

        torch.save(embedding_data, embedding_file_path)
        print(f"Saved embedding to {embedding_file_path}")

    def _check_embedding_checkpoint(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path.exists():
            print(f'Loading checkpoint from {self.checkpoint_path}')
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Checkpoint file is corrupted (JSON decode error). Re-creating checkpoint.")
                return self._create_checkpoint()
            
            # Basic validation: check if essential keys exist
            if not all(k in checkpoint_data for k in ['model_config', 'embeddings']) :
                print("Warning: Checkpoint file is missing core keys. Re-creating checkpoint.")
                return self._create_checkpoint()
            
            if not all(s in checkpoint_data['embeddings'] for s in ['test', 'training']):
                print("Warning: Checkpoint embeddings section is missing test/training keys. Re-creating checkpoint.")
                return self._create_checkpoint()

            if checkpoint_data.get('model_config') != self.model_config:
                print("Warning: Model config in checkpoint does not match current model config. Re-creating checkpoint.")
                return self._create_checkpoint()
            
            # Ensure the 'embeddings' for 'test' and 'training' are dictionaries
            if not isinstance(checkpoint_data['embeddings'].get('test'), dict):
                 checkpoint_data['embeddings']['test'] = {}
            if not isinstance(checkpoint_data['embeddings'].get('training'), dict):
                 checkpoint_data['embeddings']['training'] = {}
            
            # Initialize or validate 'failed_items' structure
            if 'failed_items' not in checkpoint_data:
                checkpoint_data['failed_items'] = {'test': {}, 'training': {}}
            if not isinstance(checkpoint_data['failed_items'].get('test'), dict):
                checkpoint_data['failed_items']['test'] = {}
            if not isinstance(checkpoint_data['failed_items'].get('training'), dict):
                checkpoint_data['failed_items']['training'] = {}
                 
            return checkpoint_data
        else:
            print(f'No checkpoint found. Creating new one at {self.checkpoint_path}')
            return self._create_checkpoint()

    def _create_checkpoint(self):
        # Structure:
        # {
        #     "model_config": {...},
        #     "embeddings": { # Successfully processed items
        #         "test": { "video_stem_1": "processed", ...},
        #         "training": { "video_stem_A": "processed", ...}
        #     },
        #     "failed_items": { # Items that failed
        #         "test": { "video_stem_X": "Error message", ...},
        #         "training": { "video_stem_Y": "Error message", ...}
        #     }
        # }
        checkpoint_data = {
            'model_config': self.model_config,
            'embeddings': {
                'test': {},
                'training': {}
            },
            'failed_items': {
                'test': {},
                'training': {}
            }
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        return checkpoint_data

    def _load_model_config(self, model_config_path: str):
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        return model_config
    
    