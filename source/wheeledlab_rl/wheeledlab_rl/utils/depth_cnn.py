import torch
import torch.nn as nn

class DepthCNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
    
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(),

            #nn.Conv2d(base_channels, 2, kernel_size=1, stride=1),
            #nn.ELU(),
            
            nn.AdaptiveMaxPool2d((2, 2)), 
            nn.Flatten(),
        )

        self.out_dim = base_channels * 2 * 2

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.net(x)