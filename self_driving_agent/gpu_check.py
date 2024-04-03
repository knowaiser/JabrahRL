import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA (GPU support) is available in PyTorch!")
    # Get the current device
    print("Current device:", torch.cuda.current_device())
    # Get the name of the current device
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA (GPU support) is not available in PyTorch, using CPU instead.")
