import torch

def get_device(force_cpu=False):
    """Get best available device with priority: MPS > CUDA > CPU

    Args:
        force_cpu: If True, force CPU usage regardless of GPU availability
    """
    if force_cpu:
        return torch.device("cpu"), False
    if torch.backends.mps.is_available():
        return torch.device("mps"), True
    elif torch.cuda.is_available():
        return torch.device("cuda"), True
    else:
        return torch.device("cpu"), False
