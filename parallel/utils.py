import os
import torch
import torch.distributed as dist
import shutil


def init_distributed_mode(args):
    """
    Initialize distributed mode based on environment variables.

    Args:
        args: Command-line arguments with distributed settings
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    # Set the device for this process
    torch.cuda.set_device(args.local_rank)
    args.device = f'cuda:{args.local_rank}'

    # Initialize the process group
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    # Synchronize all processes at this point
    dist.barrier()

    if is_main_process():
        print(
            f'Initialized process group: world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}')


def is_dist_avail_and_initialized():
    """Check if distributed environment is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """Get the number of processes in the distributed setting."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def save_on_master(state, is_best, output_dir):
    """
    Save checkpoint only from master process.

    Args:
        state: State dictionary to save
        is_best: Whether this is the best model so far
        output_dir: Directory to save the checkpoint
    """
    if is_main_process():
        torch.save(state, os.path.join(output_dir, 'checkpoint.pth'))
        if is_best:
            shutil.copyfile(
                os.path.join(output_dir, 'checkpoint.pth'),
                os.path.join(output_dir, 'model_best.pth')
            )


def init_parallel_data_loaders(dataset_dict, batch_size=32, num_workers=4):
    """
    Create data loaders optimized for parallel processing.

    Args:
        dataset_dict: Dictionary containing dataset tensors
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loading

    Returns:
        dict: Dictionary containing optimized data loaders
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Create datasets
    train_dataset = TensorDataset(dataset_dict['X_train'], dataset_dict['y_train'])
    val_dataset = TensorDataset(dataset_dict['X_val'], dataset_dict['y_val'])
    test_dataset = TensorDataset(dataset_dict['X_test'], dataset_dict['y_test'])

    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }


def setup_parallel_data_processing(dataset_path, batch_size=32, num_workers=4):
    """
    Set up parallel data processing pipeline.

    Args:
        dataset_path: Path to the dataset file
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: Dataset info dictionary and data loaders
    """
    from utils.data_utils import load_texture_dataset

    # Load dataset with parallel processing where possible
    dataset_dict = load_texture_dataset(dataset_path)

    # Create optimized data loaders
    data_loaders = init_parallel_data_loaders(
        dataset_dict,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return dataset_dict, data_loaders