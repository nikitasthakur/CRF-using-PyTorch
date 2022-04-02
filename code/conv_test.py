import torch
import torch.nn.functional as F
from conv import Conv

def get_torch_conv(X, K, padding=0):
    return F.conv2d(X, K, padding=padding)

#test case as per the pdf

#(X)
X = torch.tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], dtype=torch.float)

X = torch.unsqueeze(X, 0)
X = torch.unsqueeze(X, 0)

#(K)
K = torch.tensor([[1, 0, 1],[0, 1, 0],[1, 0, 1]], dtype=torch.float)
conv = Conv(3, kernel_tensor=K)

print("Convolution implementation Result:\n")
print(conv(X))
print("\n")

# check with PyTorch implementation
K = torch.unsqueeze(K, 0)
K = torch.unsqueeze(K, 0)

print("PyTorch implementation Result: \n")

print(get_torch_conv(X, K, 1))