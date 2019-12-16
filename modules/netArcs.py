#!/usr/bin/env python
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.featNet import FeatExtractNet,Out_Branches



class KernelConv(nn.Module):
    def __init__(self, kernel_size=5,core_bias=False):
        super(KernelConv, self).__init__()
        self.K = kernel_size
        self.core_bias = core_bias

    def split_kernel(self,core):
        ksize = self.K**2
        core = torch.split(core,ksize,dim=1)
        core = torch.stack(core,dim=1)
        return core

    def forward(self,frames,core,W):
        batch_size,N,height,width = frames.size()
        frames = frames.view(batch_size,N,height,width)
        core = self.split_kernel(core)
        pred_img_i = []
        frame_pad = F.pad(frames, [self.K//2,self.K//2,self.K//2,self.K//2])
        for i in range(self.K):
            for j in range(self.K):
                pred_img_i.append(frame_pad[..., i:i + height, j:j + width])
        pred_img_i = torch.stack(pred_img_i, dim=2)
        pred_img_i = torch.sum(core.mul(pred_img_i), dim=2, keepdim=False)
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        pred_img_i = W.mul(pred_img_i)
        return pred_img_i


class AttentionKPN(nn.Module):
    def __init__(self,nframes=5,ksize=5):
        super(AttentionKPN,self).__init__()
        self.feat   = FeatExtractNet(5)
        self.branch = Out_Branches(ksize,nframes)
        self.kconv  = KernelConv(ksize)
        self.N = nframes

    def forward(self,x):
        #import pdb;pdb.set_trace()
        feat = self.feat(x)
        kr,W = self.branch(feat)
        pimg_i = self.kconv(x,kr[:,:-self.N,...],W)
        rmap_i = (1-W).mul(kr[:,-self.N:,...])
        pimg_i += rmap_i
        pimg = torch.mean(pimg_i, dim=1, keepdim=True)\
                + torch.mean(rmap_i,dim=1,keepdim=True)
        return pimg_i,pimg



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,5,128,128).to(device)
    net = AttentionKPN()
    net.to(device)
    net.train()
    gppred,pred = net(x)
    #import pdb;pdb.set_trace()
    print(gppred.shape,pred.shape)
