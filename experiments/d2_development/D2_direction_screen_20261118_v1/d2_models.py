"""Development-only D2 candidates; frozen P2 source is imported, never edited."""
import torch
from torch import nn
from torch.nn import functional as F
from forest_unet import ForestUNet,DecoderRefinement,DoubleConv

class Refine(nn.Module):
    def __init__(self, arch, c=16):
        super().__init__();self.arch=arch
        if arch=='gated_original':
            self.body=DecoderRefinement(c,8);self.alpha=nn.Parameter(torch.zeros(()))
        else:
            if arch=='depthwise_zero':
                self.body=nn.Sequential(nn.Conv2d(2*c,c,1,bias=False),nn.BatchNorm2d(c),nn.ReLU(),nn.Conv2d(c,c,3,padding=1,groups=c,bias=False),nn.BatchNorm2d(c),nn.ReLU())
            else:self.body=DoubleConv(2*c,c)
            self.head=nn.Conv2d(c,8,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
            if arch=='spatial_zero':self.gate=nn.Conv2d(2*c,1,1)
    def forward(self,quarter,half,full,shallow):
        if self.arch=='gated_original':return self.alpha*self.body(quarter,half,full.shape[-2:])
        f=torch.cat((full,shallow),1);out=self.head(self.body(f))
        if self.arch=='spatial_zero':out=out*torch.sigmoid(self.gate(f))
        return out

class D2UNet(ForestUNet):
    def __init__(self,arch):
        super().__init__(base_channels=16,use_edge=False);self.dev_arch=arch;self.refinement=Refine(arch)
    def forward(self,x):
        e1=self.enc1(x);e2=self.enc2(self.pool(e1));e3=self.enc3(self.pool(e2));e4=self.enc4(self.pool(e3));z=self.bottleneck(self.pool(e4))
        d4=self.up4(z,e4);quarter=self.up3(d4,e3);half=self.up2(quarter,e2);full=self.up1(half,e1)
        return self.head(full)+self.refinement(quarter,half,full,e1)

def build_dev(arch):
    if arch=='A':return ForestUNet(base_channels=16,use_edge=False)
    if arch=='D_original':return ForestUNet(base_channels=16,use_edge=True)
    return D2UNet(arch)
