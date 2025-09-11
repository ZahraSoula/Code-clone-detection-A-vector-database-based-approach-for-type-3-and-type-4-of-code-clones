import torch
print(torch.version.cuda)  # Should return the CUDA version or None
print(torch.cuda.is_available())  # Should return True if CUDA is available
