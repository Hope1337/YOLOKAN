import torch
import torch.nn as nn

class CosineConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, k, 
                 stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv         = nn.Conv2d(in_ch, out_ch, k, stride, padding, dilation, groups, bias=False)

        self.conv1        = nn.Conv2d(in_ch, 1, k, stride, padding, dilation, groups, bias=False)
        self.conv1.weight = nn.Parameter(torch.ones((1, in_ch, k, k)))
        self.conv1.weight.requires_grad = False


    def get_wnorm(self):
        #[Co, Ci, k, k]
        Co, Ci, k1, k2 = self.conv.weight.shape
        weight = self.conv.weight.view(Co, -1)
        return torch.norm(weight, p=2, dim=1)
    
    def get_xnorm(self, x):
        x = torch.square(x)
        x = self.conv1(x)
        x = torch.sqrt(x + 1e-9)
        return x        

    def forward(self, x):

        #[B, Co, Ho, Wo]
        dot    = self.conv(x)

        #[Co]
        w_norm = self.get_wnorm()
        #[1, Co, 1, 1]
        w_norm = w_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        #[B, 1, Ho, Wo]
        x_norm = self.get_xnorm(x)

        return dot / (w_norm * x_norm + 1e-9)