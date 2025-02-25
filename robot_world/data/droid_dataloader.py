import tqdm
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from robot_world.data.rlds_utils import droid_dataset_transform, robomimic_transform, TorchRLDSDataset
from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics
from octo.utils.spec import ModuleSpec

tf.config.set_visible_devices([], "GPU")

def create_droid_dataloader(
    train=True,
    shuffle_buffer_size=10000,
    batch_size=128,
    balance_weights=False,
    dataset_statistics=None,
    window_size=1,
    future_action_window_size=0,
    subsample_length=100,
    skip_unlabeled=True,
    resize_primary=(360, 640),
    resize_secondary=(360, 640),
    parallel_calls=16,
    traj_transform_threads=16,
    traj_read_threads=16,
    num_workers=0,
    dataset_names=["droid_100"],
    sample_weights=[1],
    data_dir="",
    collect_fn=None,
):
    """Creates a PyTorch DataLoader for robot training data."""
    
    # Base dataset configuration
    BASE_DATASET_KWARGS = {
        "data_dir": data_dir,
        "image_obs_keys": {"primary": "exterior_image_1_left", "secondary": "exterior_image_2_left"},
        "state_obs_keys": ["cartesian_position", "gripper_position"],
        "language_key": "language_instruction",
        "norm_skip_keys": ["proprio"],
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": [True] * 10,
        "action_normalization_mask": [True] * 9 + [False],
        "standardize_fn": droid_dataset_transform,
    }
    
    # Configure filter functions
    filter_functions = [
        [ModuleSpec.create("robot_world.data.rlds_utils:filter_success")]
        if d_name in ["droid", "droid_100"] else []
        for d_name in dataset_names
    ]
    
    dataset_kwargs_list = [
        {
            "name": d_name,
            "filter_functions": f_functions,
            **BASE_DATASET_KWARGS
        }
        for d_name, f_functions in zip(dataset_names, filter_functions)
    ]
    
    if dataset_statistics is None:
        dataset_statistics = combine_dataset_statistics(
            [make_dataset_from_rlds(**dataset_kwargs, train=train)[1] 
             for dataset_kwargs in dataset_kwargs_list]
        )
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        shuffle_buffer_size=shuffle_buffer_size,
        batch_size=None,
        balance_weights=balance_weights,
        dataset_statistics=dataset_statistics,
        traj_transform_kwargs=dict(
            window_size=window_size,
            future_action_window_size=future_action_window_size,
            subsample_length=subsample_length,
            skip_unlabeled=skip_unlabeled,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs=dict(),
            resize_size=dict(
                primary=resize_primary,
                secondary=resize_secondary,
            ),
            num_parallel_calls=parallel_calls,
        ),
        traj_transform_threads=traj_transform_threads,
        traj_read_threads=traj_read_threads,
    )
    # dataset_statistics = dataset.dataset_statistics
    # sample_weights = dataset.sample_weights
    dataset = dataset.map(robomimic_transform, num_parallel_calls=parallel_calls)
    dataset.dataset_statistics = dataset_statistics
    dataset.sample_weights = sample_weights
    pytorch_dataset = TorchRLDSDataset(dataset,train=train)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collect_fn,
    )
    
    return dataloader