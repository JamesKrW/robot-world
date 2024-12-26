import torch
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict, Any, Optional
import random
tf.config.set_visible_devices([], "GPU")
class DiffusionStreamDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str = "gs://gresearch/robotics",
        split: str = "train",
        max_context_frames: int = 8,  # maximum number of context frames
        min_context_frames: int = 1,  # minimum number of context frames
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        dataset_name: str = "droid",
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        vae_encoder = None  # VAE encoder model for embedding images
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_context_frames = max_context_frames
        self.min_context_frames = min_context_frames
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.dataset_name = dataset_name
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.vae_encoder = vae_encoder
        self.current_context_frames = max_context_frames  # will be updated per batch
        
        # Initialize dataset
        self._init_dataset()
        
    def set_context_frames(self, n_frames: int):
        """Set the number of context frames for the current batch"""
        self.current_context_frames = min(max(n_frames, self.min_context_frames), self.max_context_frames)

    def _init_dataset(self) -> None:
        """Initialize the tf.data pipeline with custom splits"""
        full_dataset = tfds.load(
            self.dataset_name,
            data_dir=self.data_dir,
            split="train"
        )
        
        total_episodes = sum(1 for _ in full_dataset)
        val_size = int(total_episodes * self.validation_ratio)
        test_size = int(total_episodes * self.test_ratio)
        train_size = total_episodes - val_size - test_size
        
        if self.split == "train":
            self.dataset = full_dataset.take(train_size)
            if self.train:
                self.dataset = self.dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed,
                    reshuffle_each_iteration=True
                )
        elif self.split == "val":
            self.dataset = full_dataset.skip(train_size).take(val_size)
        elif self.split == "test":
            self.dataset = full_dataset.skip(train_size + val_size).take(test_size)
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    def _get_action(self, step) -> np.ndarray:
        """Extract and concatenate action from step"""
        return np.concatenate((
            step["action_dict"]['cartesian_position'].numpy(),
            step["action_dict"]['gripper_position'].numpy()
        ), axis=0)

    def _get_frame(self, step) -> torch.Tensor:
        """Extract and process frame from step"""
        image = step["observation"]["exterior_image_1_left"].numpy()
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        tensor = tensor / 255.0 * 2.0 - 1.0
        
        if self.vae_encoder is not None:
            with torch.no_grad():
                tensor = self.vae_encoder(tensor.unsqueeze(0)).squeeze(0)
        
        return tensor

    def _process_sequence(self, steps: list, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a sequence of steps to get frames and delta action"""
        # Get current frame (target)
        current_frame = self._get_frame(steps[frame_idx])
        
        # Calculate how many previous frames we have
        available_frames = min(frame_idx, self.current_context_frames)
        
        # Get previous frames in chronological order
        prev_frames = []
        for i in range(self.current_context_frames):
            frame_index = frame_idx - self.current_context_frames + i
            prev_frames.append(self._get_frame(steps[frame_index]))
        
        # Stack frames in temporal order (oldest to newest)
        prev_frames = torch.stack(prev_frames)  # shape: (T, C, H, W)
            
        # Calculate delta action
        current_action = self._get_action(steps[frame_idx])
        if frame_idx > 0:
            prev_action = self._get_action(steps[frame_idx - 1])
        else:
            prev_action = np.zeros_like(current_action)
        
        delta_action = current_action - prev_action
        delta_action = torch.from_numpy(delta_action).float()
        
        return prev_frames, current_frame, delta_action

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            dataset = self.dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id
            )
        else:
            dataset = self.dataset

        # For training, randomly select number of context frames per batch
        if self.train:
            self.set_context_frames(random.randint(self.min_context_frames, self.max_context_frames))

        try:
            for episode in dataset:
                steps = list(episode["steps"])
                
                # Process only sequences with enough context frames
                for i in range(self.current_context_frames, len(steps)):
                    try:
                        prev_frames, current_frame, delta_action = self._process_sequence(steps, i)
                        yield prev_frames, current_frame, delta_action
                    except Exception as e:
                        print(f"Error processing sequence: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error in dataset iteration: {e}")

def create_diffusion_dataloader(
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "gs://gresearch/robotics",
    split: str = "train",
    max_context_frames: int = 8,
    min_context_frames: int = 1,
    shuffle_buffer_size: int = 10000,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    dataset_name: str = "droid",
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    vae_encoder = None
) -> DataLoader:
    """
    Creates a DataLoader for training the diffusion model
    
    Returns:
        DataLoader: Each batch contains (prev_frames, current_frame, delta_action) where:
            - prev_frames: (B, T, C, H, W) previous frames in chronological order
            - current_frame: (B, C, H, W) current frame to predict
            - delta_action: (B, 7) delta action between prev and current frame
    """
    dataset = DiffusionStreamDataset(
        data_dir=data_dir,
        split=split,
        max_context_frames=max_context_frames,
        min_context_frames=min_context_frames,
        shuffle_buffer_size=shuffle_buffer_size,
        train=(split == "train"),
        dataset_name=dataset_name,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
        vae_encoder=vae_encoder
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

# Example usage:
if __name__ == "__main__":
    dataloader = create_diffusion_dataloader(
        batch_size=2,
        num_workers=1,
        data_dir="/home/kangrui/projects/world_model/droid-debug",
        split="train",
        max_context_frames=8,
        min_context_frames=1,
        dataset_name="droid_100"
    )
    
    # Test the dataloader
    for prev_frames, current_frame, delta_action in dataloader:
        print(f"Previous frames shape: {prev_frames.shape}")  # (B, T, C, H, W)
        print(f"Current frame shape: {current_frame.shape}")  # (B, C, H, W)
        print(f"Delta action shape: {delta_action.shape}")   # (B, 7)
        break