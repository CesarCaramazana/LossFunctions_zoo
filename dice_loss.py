class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):


        inputs = torch.nn.functional.softmax(inputs, dim=1)

        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

	dice = (2. * TP) / (2*TP + FP + FN + smooth)
	
	dice_loss = 1 - dice    

        return dice_loss
