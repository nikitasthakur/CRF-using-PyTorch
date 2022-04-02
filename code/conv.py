import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from data_loader import get_dataset

#selecting device as per GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv(nn.Module):
    
    def __init__(self, kernel_size, stride = 1, padding = True, kernel_tensor = None):
        super(Conv, self).__init__()
        self.init_params(kernel_size, stride, padding, kernel_tensor)
    
    #Function to initialize nn model params
    def init_params(self, kernel_size, stride, padding, kernel_tensor):
            
            self.padding = padding
            self.kernel_size = kernel_size
            self.stride = stride
            
            self.layer_with_padding = nn.ZeroPad2d(self.kernel_size//2)
            
            if kernel_tensor is not None:
                self.kernel = nn.Parameter(kernel_tensor)
            else:
                self.kernel = nn.Parameter(torch.randint(0, 2, (self.kernel_size, self.kernel_size), dtype=torch.float))
                
                
            
            
    #forward pass of the nn model
    
    def forward(self, X):
        
        if not self.padding:
            img = X
        else:
            img = self.layer_with_padding(X)
            
        
        end = self.kernel_size - 1
        output_dim = img.shape[-2] - end, img.shape[-1] - end
        
        conv_imgs = torch.zeros((X.shape[0], X.shape[1], output_dim[0], output_dim[1])).to(device)
        
        for l in range(img.shape[0]):
            for m in range(img.shape[1]):
                temp_img = img[l][m]
                conv_img = list()

                for i in range(img.shape[2] - end):

                    for j in range(img.shape[3] - end):

                        temp_img_view = temp_img[i : i + self.kernel_size, j : j + self.kernel_size]
                        conv_img.append(torch.sum(torch.mul(temp_img_view, self.kernel)))
                        
                conv_img = torch.stack(conv_img)
                conv_img = torch.reshape(conv_img, output_dim)
                conv_img = torch.unsqueeze(conv_img, 0)
            conv_imgs[l] = conv_img
        return conv_imgs

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """

#Test Case for Conv Forward pass

# if __name__ == "__main__":
#     image = torch.tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], dtype=torch.float)
#     image = torch.unsqueeze(image, 0)
#     image = torch.unsqueeze(image, 0)
#     kernel = torch.tensor([[1, 0, 1],[0, 1, 0],[1, 0, 1]], dtype=torch.float)
#     conv = Conv(3, kernel_tensor=kernel)
#     print(conv(image))

#     # check with PyTorch implementation
#     kernel = torch.unsqueeze(kernel, 0)
#     kernel = torch.unsqueeze(kernel, 0)
#     print(get_torch_conv(image, kernel, 1))
