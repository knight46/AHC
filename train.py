# The train file

import torch
import torch.nn as nn

from net import Mlp, BaseFeatureExtractor, DetailFeatureExtractor

nums_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


BaseTransformEncoder = nn.DataParallel(BaseFeatureExtractor()).to(device)
DetailTransformEncoder = nn.DataParallel(DetailFeatureExtractor()).to(device)

def main():
    for epoch in range(nums_epochs):
        # Training loop
        BaseTransformEncoder.train()
        DetailTransformEncoder.train()

        BaseTransformEncoder.zero_grad()
        DetailTransformEncoder.zero_grad()


if __name__ == "__main__":
    main()

