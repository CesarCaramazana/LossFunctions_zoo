# Loss Functions zoo

In this repository I recopilated some of the most commonly used loss functions for image segmentation tasks, with a brief definition of the pros and cons of each one. The code is implemented using Pytorch, so the following libraries must be imported:

import torchvision

import torch.nn as nn



So far, the loss functions available here are: 

**Cross Entropy**:
Computes the cross entropy between a prediction and a target. 
- Pros: easy to compute and to optimize.
- Cons: it doesn't handle class imbalance by definition.


**Dice Loss**:
Computes the Sørensen-Dice coefficient between a prediction (scores) and a target. The loss is defined as 1 - SDC.
- Pros: it handles class imbalanced. Constrained to [0,1]. Commonly used in segmentation tasks.
- Cons: may present some numerical inestability.


**IoU Loss**:
Computes the Jaccard coefficient between a prediction and a target. The loss is defined as 1 - JC. 
- Pros and Cons: same as Dice Loss. 


**Tversky Loss**:
It is essentially equivalent to the Dice Loss with the difference that it adds two parameters, alpha and beta, that modulate the contribution of False Negatives and False Positives. When alpha=beta=0.5, it becomes Dice Loss.
- Pros: it can be used in a scenario where we want to balance the performance of the model in terms of FP and FN. For example, in a skin detection algorithm, the cost of a False Negative should be higher than the cost of a False Positive, since it would have a major impact in people's health to miss the detection.


**Focal Loss**:
Computes the cross entropy between a prediction and a target and applies a downweighting factor (gamma) for easily classified samples.
- Pros: easy to compute and to optimize. It handles class imbalance.
- Cons: the downweighting hyperparameter gamma has to be selected by trial and error. 


**Combo Loss**:
Combines Focal loss and Tversky Loss into a single function.
- Pros: it handles highly pronounced class imbalance.
- Cons: the Tversky loss and the Focal loss values may differ in scale, leading to a case in which one dominates over the other. It has a higher computational cost than any other loss here. 
