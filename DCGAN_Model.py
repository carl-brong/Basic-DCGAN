import torch
import torch.nn as nn


class Discriminator(nn.Module):
    
    
    def __init__(self, chan_imgs, feat_d):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            #Input: N X chan_img X 64 x 64
            nn.Conv2d(
                chan_imgs,
                feat_d,
                kernel_size = 4,
                stride = 2,
                padding = 1
                ), #32 x 32
            nn.LeakyReLU(0.2),
            self._block(feat_d, feat_d*2, 4, 2, 1),
            self._block(feat_d*2, feat_d*4, 4, 2, 1),
            self._block(feat_d*4, feat_d*8, 4, 2, 1),
            nn.Conv2d(feat_d*8, 1, kernel_size = 4, stride = 2, padding = 0), # now w1x1 with 1 channel with calue to determine fake or real
            nn.Sigmoid(),    
        )
        
        
    def _block(self, in_chan, out_chan, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size,
                stride, 
                padding,
                bias = False
                ),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2),
            )
    
    
    def forward(self, x):
        return self.discriminator(x)
    
    
    
    
class Generator(nn.Module):
    
    def __init__(self, z_dim, chan_img, feat_g):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
                #Input: N X z_dim X 1x1
                self._block(z_dim, feat_g*16, 4, 1, 0), #N X f_g*16 X 4x4
                self._block(feat_g*16, feat_g*8, 4, 2, 1), #8x8
                self._block(feat_g*8, feat_g*4, 4, 2, 1), #16x16
                self._block(feat_g*4, feat_g*2, 4, 2, 1), #32x32
                nn.ConvTranspose2d(
                    feat_g*2, chan_img, kernel_size = 4, stride = 2, padding = 1 #64x64
                ),
                nn.Tanh(),
                
        )
    
    
    def _block(self, in_chan, out_chan, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_chan,
                out_chan,
                kernel_size,
                stride,
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(0.2),
        )
    
    
    def forward(self, x):
        return self.generator(x)
    

  

  
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)    
def test():
    N, in_chan, H, W = 8, 3, 64, 64,
    z_dim = 100
    x = torch.randn((N, in_chan, H, W))
    disc = Discriminator(in_chan, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_chan, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_chan, H, W)
    print("Success")
#test()
            
    