# -*- coding:utf-8 -*- 
"""
author: Reloof
date: 2022-12-24
"""
import torch.nn as nn


class Stage0(nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

    def forward(self, x):
        return self.model(x)


class Stage1(nn.Module):
    def __init__(self, in_channels=64, C1=64, S=1):
        super(Stage1, self).__init__()
        self.BTNK_1 = BTNK1(in_channels=in_channels, C1=C1, S=S)
        self.BTNK_2 = BTNK2(raw_in_channels=256)
        self.BTNK_3 = BTNK2(raw_in_channels=256)

    def forward(self, x):
        result = self.BTNK_1(x)
        result = self.BTNK_2(result)
        result = self.BTNK_3(result)
        return result


class Stage2(nn.Module):
    def __init__(self, in_channels=256, C1=128, S=2):
        super(Stage2, self).__init__()
        self.BTNK_1 = BTNK1(in_channels=in_channels, C1=C1, S=S)
        self.BTNK_2 = BTNK2(raw_in_channels=512)
        self.BTNK_3 = BTNK2(raw_in_channels=512)
        self.BTNK_4 = BTNK2(raw_in_channels=512)

    def forward(self, x):
        result = self.BTNK_1(x)
        result = self.BTNK_2(result)
        result = self.BTNK_3(result)
        result = self.BTNK_4(result)
        return result


class BTNK1(nn.Module):
    def __init__(self, in_channels, C1, S):
        super(BTNK1, self).__init__()
        raw_in_channels = C1
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, kernel_size=(1, 1), out_channels=raw_in_channels, stride=S),
            nn.BatchNorm2d(raw_in_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=raw_in_channels, kernel_size=(3, 3), out_channels=raw_in_channels, stride=1,
                      padding=1),
            nn.BatchNorm2d(raw_in_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=raw_in_channels, kernel_size=(1, 1), out_channels=4 * raw_in_channels, stride=1),
            nn.BatchNorm2d(4 * raw_in_channels)
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, kernel_size=(1, 1), out_channels=4 * raw_in_channels, stride=S),
            nn.BatchNorm2d(4 * raw_in_channels)
        )

        self.relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x):
        result = self.relu((self.model1(x) + self.model2(x)))
        return result


class BTNK2(nn.Module):
    def __init__(self, raw_in_channels):
        super(BTNK2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=raw_in_channels, kernel_size=(1, 1), out_channels=raw_in_channels // 4, stride=1),
            nn.BatchNorm2d(raw_in_channels // 4),
            nn.ReLU(),

            nn.Conv2d(in_channels=raw_in_channels // 4, kernel_size=(3, 3), out_channels=raw_in_channels // 4, stride=1,
                      padding=1),
            nn.BatchNorm2d(raw_in_channels // 4),
            nn.ReLU(),

            nn.Conv2d(in_channels=raw_in_channels // 4, kernel_size=(1, 1), out_channels=raw_in_channels, stride=1),
            nn.BatchNorm2d(raw_in_channels)
        )

        self.relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x):
        result = self.model(x)
        result = self.relu(x + result)
        return result


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.Stage0 = Stage0()
        self.Stage1 = Stage1()
        self.Stage2 = Stage2()

        self.classify = nn.Sequential(
            # nn.BatchNorm2d(2048),
            nn.MaxPool2d(28, 28),
            nn.Flatten(),
            nn.Linear(512, 8),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        result = self.Stage0(x)
        result = self.Stage1(result)
        result = self.Stage2(result)
        result = self.classify(result)
        return result
