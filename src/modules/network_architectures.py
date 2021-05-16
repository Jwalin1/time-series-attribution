# for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

  def __init__(self, num_classes:int=2, inp_channels:int=3) -> None:
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(    # inp_len = 50             np.floor((Lin + 2*padding - kernel_size)/stride + 1)
      nn.Conv1d(inp_channels, 64, kernel_size=11, stride=4, padding=2),   # 50 -> 11 
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),    # 11 -> 5
      nn.Conv1d(64, 192, kernel_size=5, padding=2), # same
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2),  # 5 -> 2
      nn.Conv1d(192, 384, kernel_size=3, padding=1),  # same
      nn.ReLU(inplace=True),
      nn.Conv1d(384, 256, kernel_size=3, padding=1),  # same
      nn.ReLU(inplace=True),
      nn.Conv1d(256, 256, kernel_size=3, padding=1),  # same
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=2, stride=2),    # 2 -> 1
    )
    self.avgpool = nn.AdaptiveAvgPool1d(6)
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
    )
    # placeholder for the gradients
    self.gradients = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)

    # register the hook
    if x.requires_grad:
      h = x.register_hook(self.activations_hook)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  # hook for the gradients of the activations
  def activations_hook(self, grad):
    self.gradients = grad

  # method for the gradient extraction
  def get_activations_gradient(self):
    return self.gradients
  
  # method for the activation exctraction
  def get_activations(self, x):
    return self.features(x)