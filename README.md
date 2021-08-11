# LossFunctions_zoo

In this repository I recopilated some of the most commonly used loss functions for image segmentation tasks, with a brief definition of the pros and cons of each one. The code is implemented using Pytorch, so the following libraries must be imported:

import torchvision
import torch.nn as nn

So far, the loss functions implemented are: 

**Cross Entropy**:
Computes the cross entropy between a prediction and a target. 
- Pros: easy to compute and to optimize.
- Cons: it doesn't handle class imbalance by definition.
