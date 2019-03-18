"""Implementation of original Disco-GAN architecture"""
import os
import numpy as np
import fastai
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Vinay Kumar"
__copyright__ = "Vinay Kumar @2019"
__license__ = "MIT"


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64*2, 1, 2, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(64*2)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)
        self.conv3 = nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(64*4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(64*8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(64*8, 1, 4, 1, 0, bias=True)


    def forward(self, input):
        out_conv1 = self.conv1(input)
        out_relu1 = self.relu1(out_conv1)
        out_conv2 = self.conv2(out_relu1)
        out_bn2 = self.bn2(out_conv2)
        out_relu2 = self.relu2(out_bn2)
        out_conv3 = self.conv3(out_relu2)
        out_bn3 = self.bn3(out_conv3)
        out_relu3 = self.relu3(out_bn3)
        out_conv4 = self.conv4(out_relu3)
        out_bn4 = self.bn4(out_conv4)
        out_relu4 = self.relu4(out_bn4)
        out_conv5 = self.conv5(out_relu4)

        return torch.sigmoid(out_conv5), [out_relu1, out_relu2, out_relu3, out_relu4]


class Generator(torch.nn.Module):
    def __init__(self, extra_layers=False):
        super(Generator, self).__init__()

        if extra_layers == True:
            self.generator = nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1, bias=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 64*2, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*8),
                    nn.LeakyReLU(0.2, bias=True),
                    nn.Conv2d(64*8, 100, 4, 1, 0, bias=True),
                    nn.BatchNorm2d(100),
                    nn.LeakyReLU(0.2, bias=True),
                    nn.ConvTranspose2d(100, 64*8, 4, 1, 0, bias=True),
                    nn.BatchNorm2d(64*8),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*4),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=True),
                    nn.Sigmoid()
                )

        elif extra_layers == False:
            self.generator = nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 64*2, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*4),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64*2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=True),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=True),
                    nn.Sigmoid()
                )


    def forward(self, input):
       return self.generator(input)

