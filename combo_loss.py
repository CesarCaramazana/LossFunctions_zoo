class ComboLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ComboLoss, self).__init__()
        self.kwargs = kwargs       

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, epsilon=1, gamma=2):
      """
        inputs: raw output predictions (without activation). Shape: [batch, num_classes, h, w]
        targets: ground truth labels (one-hot encoded). Shape: [batch, num_classes, h, w]
	alpha: False Positives weighting coefficient
	beta: False Negatives weighting coefficient
	epsilon, gamma: Focal Loss parameters

      """
        
        #Tversky Loss (Dice loss when alpha=beta=1)---
        pred = torch.nn.functional.softmax(inputs, dim=1)

        #flatten label and prediction tensors
        pred = pred.contiguous().view(-1)
        labels = targets.contiguous().view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (pred * labels).sum()    
        FP = ((1-labels) * pred).sum()
        FN = (labels * (1-pred)).sum()

	Tversky = (2*TP + smooth) / (2*TP + alpha*FP + beta*FN + smooth) 
	tversky_loss = 1 - Tversky


        #Focal Cross Entropy Loss---
        _, targets = torch.max(targets, dim=1)
    

        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = epsilon* (1-pt)**gamma * BCE_loss

        F_loss = F_loss.mean()

	

        return tversky_loss + F_loss