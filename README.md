# Loss Functions zoo

In this repository I recopilated some of the most commonly used loss functions for image segmentation tasks, with a brief definition of the pros and cons of each one. The code is implemented using Pytorch. 

So far, the loss functions available here are: 

**Cross Entropy**:
Computes the cross entropy between a prediction and a target. 
- Pros: easy to compute and to optimize.
- Cons: it doesn't handle class imbalance by definition.

File: <code>cross_entropy.py</code>

**Dice Loss**:
Computes the Sørensen-Dice coefficient between a prediction (scores) and a target. The loss is defined as 1 - SDC.
- Pros: it handles class imbalanced. Constrained to [0,1]. Commonly used in segmentation tasks.
- Cons: may present some numerical inestability.

<img src="https://github.com/CesarCaramazana/LossFunctions_zoo/blob/main/images/dice.png" width="450px">

File: <code>dice_loss.py</code>

**IoU Loss**:
Computes the Jaccard coefficient between a prediction and a target. The loss is defined as 1 - JC. 
- Pros and Cons: same as Dice Loss. 

<img src="https://github.com/CesarCaramazana/LossFunctions_zoo/blob/main/images/iou.png" width="320px">

Image source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

File: <code>iou_loss.py</code>

**Tversky Loss**:
It is essentially equivalent to the Dice Loss with the difference that it adds two parameters, alpha and beta, that modulate the contribution of False Negatives and False Positives. When alpha=beta=0.5, it becomes Dice Loss.
- Pros: it can be used in a scenario where we want to balance the performance of the model in terms of FP and FN. For example, in a skin detection algorithm, the cost of a False Negative should be higher than the cost of a False Positive, since it would have a major impact in people's health to miss the detection.

<img src="https://github.com/CesarCaramazana/LossFunctions_zoo/blob/main/images/tversky.png" width="450px">

File: <code>tversky_loss.py</code>

**Focal Loss**:
Computes the cross entropy between a prediction and a target and applies a downweighting factor (gamma) for easily classified samples.
- Pros: easy to compute and to optimize. It handles class imbalance.
- Cons: the downweighting hyperparameter gamma has to be selected by trial and error. 

<img src="https://github.com/CesarCaramazana/LossFunctions_zoo/blob/main/images/focal.png" width="500px">

Image source: https://doi.org/10.1109/ICCV.2017.324

File: <code>focal_loss.py</code>

**Combo Loss**:
Combines Focal loss and Tversky Loss into a single function.
- Pros: it handles highly pronounced class imbalance.
- Cons: the Tversky loss and the Focal loss values may differ in scale, leading to a case in which one dominates over the other. It has a higher computational cost than any other loss here. 

File: <code>combo_loss.py</code>
