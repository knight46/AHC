# The train file

import torch
import torch.nn as nn
from astropy.visualization import BaseTransform

from net import Mlp, BaseFeatureExtractor, DetailFeatureExtractor

nums_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BaseTransformEncoder = nn.DataParallel(BaseFeatureExtractor()).to(device)

def main():
    for epoch in range(nums_epochs):
        # Training loop


if __name__ == "__main__":
    main()

