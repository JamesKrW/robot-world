import torch
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict, Any, Optional
import albumentations as A
# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")
class DROIDImageTransform:
    """Handles image transformations with augmentation."""
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        train: bool = True,
        image_aug: bool = True
    ):
        aug_transforms = []
        if image_aug and train:
            aug_transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RandomResizedCrop(
                    height=image_size[0],
                    width=image_size[1],
                    scale=(0.9, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5
                )
            ])

        self.transform = A.Compose([
            *aug_transforms,
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        transformed = self.transform(image=image)['image']
        return torch.from_numpy(transformed).permute(2, 0, 1).float()

class DROIDStreamDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str = "gs://gresearch/robotics",
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        image_aug: bool = True,
        dataset_name: str = "droid_100"
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.dataset_name = dataset_name
        # Initialize image transform
        self.image_transform = DROIDImageTransform(
            image_size=image_size,
            train=train,
            image_aug=image_aug
        )
        
        # Initialize dataset
        self._init_dataset()

    def _init_dataset(self) -> None:
        """Initialize the tf.data pipeline with optimizations"""
        # Load the dataset
        self.dataset = tfds.load(
            self.dataset_name,
            data_dir=self.data_dir,
            split=self.split
        )
        
        if self.train:
            self.dataset = self.dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                reshuffle_each_iteration=True
            )
        
        # Apply prefetching
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    def _process_step(self, step) -> torch.Tensor:
        """Process a single step from an episode"""
        # Convert TensorFlow tensors to NumPy arrays
        concat_image = np.concatenate((
            step["observation"]["exterior_image_1_left"].numpy(),
            step["observation"]["wrist_image_left"].numpy(),
            step["observation"]["exterior_image_2_left"].numpy(),
        ), axis=1)
        
        return self.image_transform(concat_image)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # Shard the dataset if using multiple workers
            dataset = self.dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id
            )
        else:
            dataset = self.dataset

        try:
            # Iterate through episodes
            for episode in dataset:
                # Iterate through steps in each episode
                for step in episode["steps"]:
                    try:
                        processed_image = self._process_step(step)
                        yield processed_image
                    except Exception as e:
                        print(f"Error processing step: {e}")
                        continue
        except Exception as e:
            print(f"Error in dataset iteration: {e}")

def create_droid_dataloader(
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "gs://gresearch/robotics",
    split: str = "train",
    image_size: Tuple[int, int] = (256, 256),
    shuffle_buffer_size: int = 10000,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    image_aug: bool = True,
) -> DataLoader:
    """
    Creates an optimized DataLoader for the DROID dataset
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        data_dir: Dataset directory
        split: Dataset split
        image_size: Target image size
        shuffle_buffer_size: Size of shuffle buffer
        pin_memory: Whether to pin memory
        prefetch_factor: Number of batches to prefetch
        image_aug: Whether to use image augmentation
    
    Returns:
        DataLoader: Optimized PyTorch DataLoader
    """
    dataset = DROIDStreamDataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size,
        shuffle_buffer_size=shuffle_buffer_size,
        train=(split == "train"),
        image_aug=image_aug,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

# Test code
def test_dataloader():
    print("Creating dataloader...")
    dataloader = create_droid_dataloader(
        batch_size=2,
        num_workers=1,
        image_size=(256, 256),
        shuffle_buffer_size=1000,
        data_dir="/home/kangrui/projects/world_model/droid-debug",
    )
    
    print("Attempting to load a batch...")
    try:
        batch = next(iter(dataloader))
        print(f"Successfully loaded batch with shape: {batch.shape}")
        
        # Test batch properties
        print(f"Batch min value: {batch.min()}")
        print(f"Batch max value: {batch.max()}")
        print(f"Batch mean value: {batch.mean()}")
        # test if on gpu
        print(f"Batch on GPU: {batch.is_cuda}")
        
        return True
    except Exception as e:
        print(f"Error loading batch: {e}")
        return False

if __name__ == "__main__":
    test_dataloader()