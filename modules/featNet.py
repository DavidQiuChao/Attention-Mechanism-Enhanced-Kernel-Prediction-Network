#!/usr/bin/env python
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cbam import *
#from cbam import *


class Basic_Block(nn.Module):
    def __init__(self,In,Out,ks,std,pad):
        super(Basic_Block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(In,Out,ks,std,pad),
                nn.ReLU(inplace=True),
                nn.Conv2d(Out,Out,ks,std,pad),
                nn.ReLU(inplace=True),
                nn.Conv2d(Out,Out,ks,std,pad),
                nn.ReLU(inplace=True)
            )

    def forward(self,x):
        return self.conv(x)


class Attention_Block(nn.Module):
    def __init__(self,In,Out,ks,std,pad):
        super(Attention_Block,self).__init__()
        self.bb = Basic_Block(In,Out,ks,std,pad)
        self.cbam = CBAM(Out,16)

    def forward(self,x):
        out = self.bb(x)
        out = self.cbam(out)
        return out


class Up_Skip_Block(nn.Module):
    def __init__(self,In,Out):
        super(Up_Skip_Block,self).__init__()
        self.up = nn.Upsample(scale_factor=2,\
                mode='bilinear',align_corners=True)
        self.ab = Attention_Block(In,Out,3,1,1)

    def forward(self,x,skip):
        out = torch.cat([self.up(x),skip],1)
        out = self.ab(out)
        return out


class Out_Up_Skip_Block(nn.Module):
    def __init__(self,In):
        super(Out_Up_Skip_Block,self).__init__()
        self.up = nn.Upsample(scale_factor=2,\
                mode='bilinear',align_corners=True)

    def forward(self,x,skip):
        out = torch.cat([self.up(x),skip],1)
        return out


class FeatExtractNet(nn.Module):
    def __init__(self,In):
        super(FeatExtractNet,self).__init__()
        self.bb1 = Basic_Block(In,64,3,1,1)
        self.bb2 = Basic_Block(64,128,3,1,1)
        self.bb3 = Basic_Block(128,256,3,1,1)
        self.bb4 = Basic_Block(256,512,3,1,1)
        self.bb5 = Basic_Block(512,512,3,1,1)
        self.up1 = Up_Skip_Block(1024,512)
        self.up2 = Up_Skip_Block(768,256)
        self.up3 = Up_Skip_Block(384,128)
        self.outup = Out_Up_Skip_Block(192)

    def forward(self,x):
        feat1  = self.bb1(x)
        feat2  = self.bb2(F.avg_pool2d(feat1,kernel_size=2,stride=2))
        feat3  = self.bb3(F.avg_pool2d(feat2,kernel_size=2,stride=2))
        feat4  = self.bb4(F.avg_pool2d(feat3,kernel_size=2,stride=2))
        feat5  = self.bb5(F.avg_pool2d(feat4,kernel_size=2,stride=2))
        feat6  = self.up1(feat5,feat4)
        feat7  = self.up2(feat6,feat3)
        feat8  = self.up3(feat7,feat2)
        outfeat  = self.outup(feat8,feat1)
        return outfeat


class Out_Branches(nn.Module):
    def __init__(self,Ks,N):
        super(Out_Branches,self).__init__()
        self.bb = Basic_Block(192,192,3,1,1)
        abOut = (Ks**2+1)*N
        self.ab = Attention_Block(192,192,3,1,1)
        self.bhk = nn.Conv2d(192,abOut,1,1)
        self.bhw = nn.Conv2d(192,N,1,1)

    def forward(self,x):
        kr = self.ab(x)
        kr = self.bhk(kr)
        W = self.bb(x)
        W = torch.sigmoid(self.bhw(W))
        return kr,W



if __name__ == '__main__':
    import os
    import time
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,1,128,128).to(device)
    net = FeatExtractNet(1)
    net.to(device)
    net.eval()
    t0 = time.time()
    ofeat = net(x)
    t1 = time.time()
    print('cost %f'%((t1-t0)))
    print(ofeat.shape)
