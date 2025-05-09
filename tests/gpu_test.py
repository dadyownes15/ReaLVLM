import torch

def test_cuda_available():
    print("Checking for CUDA availability...")
    
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Optional: run a small tensor operation on the GPU
        try:
            x = torch.rand(1000, 1000).to('cuda')
            y = torch.mm(x, x)
            print("✅ Tensor operation succeeded on the GPU.")
        except Exception as e:
            print("⚠️ Tensor operation failed:", e)
    else:
        print("❌ CUDA is not available on this system.")

if __name__ == "__main__":
    test_cuda_available()
