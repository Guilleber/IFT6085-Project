import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    def __init__(self, p=0.3):
        super(Dropout, self).__init__()
        self.p = p
         
    def forward(self, input):
        if self.training:
            mask = (t.FloatTensor(input.size()).uniform_() > self.p).float()
            if input.is_cuda:
                mask = mask.cuda()
            return 1/(1-self.p) * input * mask
        else:
            return input