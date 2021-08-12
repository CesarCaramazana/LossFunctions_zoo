class FocalLoss(nn.Module):
  def __init__(self, alpha=1, gamma=2):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma


  def forward(self, inputs, targets):
      """
        inputs: raw output predictions (without activation). Shape: [batch, num_classes, h, w]
        targets: ground truth labels (one-hot encoded). Shape: [batch, num_classes, h, w]
      """
    _, targets = torch.max(targets, dim=1) #Undo one-hot


    BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

    pt = torch.exp(-BCE_loss)
    F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

    return torch.mean(F_loss)
