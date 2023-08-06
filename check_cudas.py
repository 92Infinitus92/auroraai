import torch

def check_cuda_devices():
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices found. Make sure you have the appropriate GPU drivers and CUDA toolkit installed.")

if __name__ == "__main__":
    # Set the desired GPU device (e.g., 0 for the first GPU)
    device = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")

    # Move the tensors and models to the selected device
    check_cuda_devices()
