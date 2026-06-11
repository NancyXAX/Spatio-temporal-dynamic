import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()

    for i in range(device_count):
        print(f"GPU{i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. No GPU devices found.")