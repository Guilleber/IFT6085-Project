import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, input_size, e=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.input_size = input_size
        self.e = e
        self.momentum = momentum
        self.gamma = nn.Parameter(t.rand(input_size, requires_grad = True))
        self.beta = nn.Parameter(t.zeros(input_size, requires_grad = True))
        #self.gamma.requires_grad = True
        #self.beta.requires_grad = True
        self.register_buffer('estimate_E', t.zeros(input_size, requires_grad = False))
        self.register_buffer('estimate_Var', t.zeros(input_size, requires_grad = False))


    def forward(self, input):
        if self.training:
            E = t.mean(input, 0)
            Var = t.std(input, 0)**2
            with t.no_grad():
                self.estimate_E = (1-self.momentum)*self.estimate_E + self.momentum*E 
                self.estimate_Var = (1-self.momentum)*self.estimate_Var + self.momentum*Var
            out = (input - E)/t.sqrt(Var + self.e)
            out = self.gamma*out + self.beta
        else:
            out = (input - self.estimate_E)/t.sqrt(self.estimate_Var + self.e)
            out = self.gamma*out + self.beta
        return out