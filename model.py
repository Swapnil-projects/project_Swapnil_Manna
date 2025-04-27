import torch
import torch.nn as nn
import torch.nn.functional as F

class DeblurModel(nn.Module):
    def __init__(self):
        super(OptimizedMediumSmallUNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.enc1 = conv_block(3, 48)      
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(48, 96)     
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(96, 192)    
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(192, 384)   
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            conv_block(384, 512),
            nn.Dropout(0.3)
        )

        self.up4 = up_block(512, 384)      
        self.dec4 = conv_block(768, 384)

        self.up3 = up_block(384, 192)      
        self.dec3 = conv_block(384, 192)

        self.up2 = up_block(192, 96)       
        self.dec2 = conv_block(192, 96)

        self.up1 = up_block(96, 48)        
        self.dec1 = conv_block(96, 48)

        self.final = nn.Conv2d(48, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.tanh(self.final(d1))
