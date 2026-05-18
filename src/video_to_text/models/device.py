import torch
import psutil

def get_device_type() -> str:
    """Detect the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def empty_cache() -> None:
    """Safely empty cache based on the active device."""
    import gc
    gc.collect()
    
    device = get_device_type()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == "mps":
        torch.mps.empty_cache()

def get_system_ram_gb() -> float:
    """Return total system RAM in GB."""
    ram_info = psutil.virtual_memory()
    return ram_info.total / (1024 ** 3)
