#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import numpy


class RNNModel():
    def __init__(self, args):
        self.title_c = nn.LSTM(args.title_c_input, args.title_c_hidden, args.title_c_layers, bidirrectional=True)
        self.title_w = nn.LSTM(args.title_w_input, args.title_w_hidden, args.title_w_layers, bidirrectional=True)
        self.desc_c = nn.LSTM(args.desc_c_input, args.desc_c_hidden, args.desc_c_layers, bidirrectional=True)
        self.desc_w = nn.LSTM(args.desc_w_input, args.desc_w_hidden, args.desc_w_layers, bidirrectional=True)
        self.fc1 = nn.Linear()

    def forward(self, t_c, t_w, d_c, d_w):
        out_t_c, _ = self.title_c(t_c)
        out_t_w, _ = self.title_w(t_c)
        out_d_c, _ = self.desc_c(d_c)
        out_d_w, _= self.desc_w(d_w)

        out_t_c = out_t_c
        out_t_w = out_t_w
        out_d_c = out_d_c
        out_d_w = out_d_w

        out_t_c = out_t_c.view(out_t_c.shape[1], -1)
        out_t_w = out_t_w.view(out_t_w.shape[1], -1)
        out_d_c = out_d_c.view(out_d_c.shape[1], -1)
        out_d_w = out_d_w.view(out_d_w.shape[1], -1)

        out = torch.cat((out_t_c, out_t_w, out_d_c, out_d_w), 1)





