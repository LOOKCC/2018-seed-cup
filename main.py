#!/usr/bin/env python
# coding=utf-8
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Whether to test directly (default is training)')
    parser.add_argument('--root', default='../../data', help='Path to dataset (default="data/")')
    parser.add_argument('--model_path', default='checkpoints/fcn.pth', help='Path to save model to save (default="checkpoints/crnn.pth")')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default=4)')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size (default=64)')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs to train for (default=50)')
    parser.add_argument('--check_epoch', type=int, default=10, help='Epoch to save and test (default=10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Critic (default=0.1)')
    parser.add_argument('--', type=int, default=2, help='Resize origin image to 1/factor^2 (default=2)')

    parser.add_argument('--title_c_input', type=int, default=0, help='Number of title characters input  (default=0)')
    parser.add_argument('--title_c_hidden', type=int, default=0, help='Number of title characters hidden output (default=0)')
    parser.add_argument('--title_c_layers', type=int, default=0, help='Number of title characters hidden layers (default=0)')
    parser.add_argument('--title_w_input', type=int, default=0, help='Number of title word input (default=0)')
    parser.add_argument('--title_w_hidden', type=int, default=0, help='Number of title word hidden output (default=0)')
    parser.add_argument('--title_w_layers', type=int, default=0, help='Number of title word hidden layers (default=0)')
    parser.add_argument('--dsec_c_input', type=int, default=0, help='Number of dsecription characters input (default=0)')
    parser.add_argument('--desc_c_hidden', type=int, default=0, help='Number of dsecription characters hidden output (default=0)')
    parser.add_argument('--desc_c_layers', type=int, default=0, help='Number of dsecription characters hidden layers (default=0)')
    parser.add_argument('--desc_w_input', type=int, default=0, help='Number of dsecription word input (default=0)')
    parser.add_argument('--desc_w_hidden', type=int, default=0, help='Number of dsecription word hidden output (default=0)')
    parser.add_argument('--desc_w_layers', type=int, default=, help='Number of dsecription word hidden layers (default=0)')
    opt = parser.parse_args()
    print(opt)
