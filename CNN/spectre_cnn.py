import torch.nn as nn
from config import Config

class SpectreCNN(nn.Module):
    """
    SpectreCNN Model.
    Args:
        in_channels: An integer representing the input embedding channels.
        out_channels: An integer representing the output of the convolution.
        kernel_size: A tuple of integers representing the size of convolving kernel. 
        stride: A tuple of integers representing convolution stride. 
        padding: A tuple of integers that represent sides of embedding.
        pool_size: A tuple of integers representing the H x W output of embedding size.
        in_features: An integer representing the size of input embedding sample.
        out_features: An integer representing the size of output embedding sample.
        drop_out_prob: A float representing a probability of an element to be zeroed.
        labe_num: An integer representing the number of target labels.
    """
    
    def __init__(self):
        super(SpectreCNN, self).__init__()
        self.CONFIG = Config()

        self.l1 = nn.Sequential(
            nn.Conv2d(self.CONFIG.IN_CHANNELS, self.CONFIG.OUT_CHANNELS, self.CONFIG.KERNEL_SIZE, self.CONFIG.STRIDE, self.CONFIG.PADDING[0]),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.CONFIG.MAX_POOL)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(self.CONFIG.IN_CHANNELS, self.CONFIG.OUT_CHANNELS, self.CONFIG.KERNEL_SIZE, self.CONFIG.STRIDE, self.CONFIG.PADDING[1]),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.CONFIG.MAX_POOL)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(self.CONFIG.IN_CHANNELS, self.CONFIG.OUT_CHANNELS, self.CONFIG.KERNEL_SIZE, self.CONFIG.STRIDE, self.CONFIG.PADDING[2]),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(self.CONFIG.MAX_POOL)
        )

        self.fc1 = nn.Linear(self.CONFIG.IN_FEATURES * self.CONFIG.IN_FEATURES, self.CONFIG.OUT_FEATURES)
        self.fc2 = nn.Linear(self.CONFIG.OUT_FEATURES, self.CONFIG.LABEL_NUM)
        
        self.dropout = nn.Dropout(self.CONFIG.DROP_OUT_PROB)
        self.flatten = nn.Flatten()
        
        self.model = nn.Sequential(
            self.l1,
            self.l2,
            self.l3,
            self.flatten,
            self.dropout,
            self.fc1,
            self.dropout,
            self.fc2,
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        return self.model(embeddings)