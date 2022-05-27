# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from function import trainingHook

class TrainingHook(nn.Module):
    def __init__(self, label_features, dim_hook, train_mode):
        super(TrainingHook, self).__init__()
        self.train_mode = train_mode

        if self.train_mode in ["TP"]:
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()
        else:
            self.fixed_fb_weights = None

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        return trainingHook(input, labels, y, self.fixed_fb_weights, self.train_mode if (self.train_mode != "FA") else "BP") #FA is handled in FA_wrapper, not in TrainingHook

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.train_mode + ')'
