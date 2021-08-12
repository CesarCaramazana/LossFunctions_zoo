class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
	
      	"""
        inputs: raw output predictions (without activation). Shape: [batch, num_classes, h, w]
        targets: ground truth labels (one-hot encoded). Shape: [batch, num_classes, h, w]
      	"""
     
        inputs = torch.nn.functional.softmax(inputs, dim=1)


        #flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        A_sum = (inputs * inputs).sum()
        B_sum = (targets * targets).sum()

        iou = intersection / (A_sum + B_sum - intersection + smooth)
               
        iou_loss = 1 - iou

        return iou_loss