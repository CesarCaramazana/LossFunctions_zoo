class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
	"""
        inputs: raw output predictions (without activation). Shape: [batch, num_classes, h, w]
        targets: ground truth labels (one-hot encoded). Shape: [batch, num_classes, h, w]
	alpha: False Positives weighting coefficient
	beta: False Negatives weighting coefficient
        """
        
        inputs = torch.nn.functional.softmax(inputs, dim=1)


        #flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

                

        Tversky = (2*TP + smooth) / (2*TP + alpha*FP + beta*FN + smooth)  

   
        return 1 - Tversky