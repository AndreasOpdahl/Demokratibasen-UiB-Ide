# test_gpu.py
import torch, time
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(4096, 4096, device="cuda")
y = torch.mm(x, x.t())
torch.cuda.synchronize()
print("OK:", y.shape, "on", torch.cuda.get_device_name(0))
time.sleep(2)  # keep job alive briefly
