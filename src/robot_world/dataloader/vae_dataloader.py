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
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Transform image
        transformed = self.transform(image=image)['image']
        # Convert to tensor, scale to [0,1], then to [-1,1]
        tensor = torch.from_numpy(transformed).permute(2, 0, 1).float()
        tensor = tensor / 255.0  # to [0,1]
        tensor = tensor * 2.0 - 1.0  # to [-1,1]
        return tensor

class DROIDStreamDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str = "gs://gresearch/robotics",
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        image_aug: bool = True,
        dataset_name: str = "droid",
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split  # now can be "train", "val", or "test"
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.dataset_name = dataset_name
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Initialize image transform
        self.image_transform = DROIDImageTransform(
            image_size=image_size,
            train=train,
            image_aug=image_aug
        )
        
        # Initialize dataset
        self._init_dataset()

    def _init_dataset(self) -> None:
        """Initialize the tf.data pipeline with custom splits"""
        # Load the full dataset
        full_dataset = tfds.load(
            self.dataset_name,
            data_dir=self.data_dir,
            split="train"  # Original split is always "train"
        )
        
        # Get total number of episodes
        total_episodes = sum(1 for _ in full_dataset)
        
        # Calculate split boundaries
        val_size = int(total_episodes * self.validation_ratio)
        test_size = int(total_episodes * self.test_ratio)
        train_size = total_episodes - val_size - test_size
        
        # Create deterministic splits using take/skip
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
        
        # Apply prefetching
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    def _process_step(self, step) -> torch.Tensor:
        """Process a single step from an episode"""
        # Convert TensorFlow tensors to NumPy arrays
        concat_image = np.concatenate((
            step["observation"]["exterior_image_1_left"].numpy(),
            # step["observation"]["wrist_image_left"].numpy(),
            # step["observation"]["exterior_image_2_left"].numpy(),
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
    image_size: Tuple[int, int] = (180, 320),
    shuffle_buffer_size: int = 10000,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    image_aug: bool = True,
    dataset_name: str = "droid",
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> DataLoader:
    """
    Creates an optimized DataLoader for the DROID dataset with custom splits
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        data_dir: Dataset directory
        split: One of ["train", "val", "test"]
        image_size: Target image size
        shuffle_buffer_size: Size of shuffle buffer
        pin_memory: Whether to pin memory
        prefetch_factor: Number of batches to prefetch
        image_aug: Whether to use image augmentation
        validation_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        DataLoader: Optimized PyTorch DataLoader
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Split must be one of ['train', 'val', 'test'], got {split}")
    
    dataset = DROIDStreamDataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size,
        shuffle_buffer_size=shuffle_buffer_size,
        train=(split == "train"),
        image_aug=image_aug and (split == "train"),  # Only use augmentation for training
        dataset_name=dataset_name,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed
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
    def save_image(batch, split="debug", idx=0):
        """Save image with proper color space and range handling"""
        img = batch[idx].permute(1, 2, 0)
        img = torch.clamp((img + 1) / 2, 0, 1)
        img = (img * 255).byte().cpu().numpy()
        import cv2
        cv2.imwrite(f"{split}_image.jpg", img)
        return img
    
    def test_all_splits():
        data_dir = "/home/kangrui/projects/world_model/droid-debug"
        batch_size = 2
        
        # Create dataloaders for all splits
        splits = ["train", "val", "test"]
        for split in splits:
            print(f"\nTesting {split} split...")
            dataloader = create_droid_dataloader(
                batch_size=batch_size,
                num_workers=1,
                data_dir=data_dir,
                split=split,
                dataset_name="droid_100",
                validation_ratio=0.1,
                test_ratio=0.1,
                seed=42
            )
            
            try:
                batch = next(iter(dataloader))
                print(f"Successfully loaded batch with shape: {batch.shape}")
               
                # transform back
                save_image(batch, split=split)
            except Exception as e:
                print(f"Error loading batch: {e}")
    test_all_splits()