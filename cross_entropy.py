class CrossEntropy(nn.Module):
  def __init__(self):
    super(CrossEntropy,self).__init__()

  def forward(self, inputs, targets):
      """
        inputs: raw output predictions (without activation). Shape: [batch, num_classes, h, w]
        targets: ground truth labels (one-hot encoded). Shape: [batch, num_classes, h, w]
      """
    _, labels = torch.max(targets, dim=1) #Undo one-hot

    CE = nn.CrossEntropyLoss(reduction='mean')(inputs, labels)

    return CE
