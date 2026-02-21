"""GPU memory monitoring utilities."""
import torch


def print_gpu_info():
    """Print current GPU information and memory usage."""
    if not torch.cuda.is_available():
        return
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    print("\n" + "=" * 70)
    print("GPU Information & Memory Status")
    print("=" * 70)
    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"\nMemory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("=" * 70 + "\n")


def get_gpu_memory_stats():
    """Get current GPU memory stats as dictionary."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
