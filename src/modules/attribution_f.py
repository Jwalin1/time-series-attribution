import numpy as np
import matplotlib.pyplot as plt

# for neural network
import torch

# for attribution
from captum.attr import Saliency, IntegratedGradients, InputXGradient, GuidedBackprop, GuidedGradCam, Lime
from captum.attr import LayerGradCam, LayerAttribution

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def applyMethod(method, model, samples):
  model.eval()
  maps = []
  for sample in samples:
    
    sample = torch.from_numpy(sample).to(device).unsqueeze(0)
    sample = sample.float().requires_grad_(True)
    target = model(sample).argmax(dim=1)[0]
    if method in ["LayerGradCam", "GuidedGradCam"]:
      interpreter = globals()[method](model, model.features[-3])
    else:
      interpreter = globals()[method](model)
      
    attribution = interpreter.attribute(sample, target=target)
    if method in ["LayerGradCam", "GuidedGradCam"]:
      attribution = LayerAttribution.interpolate(attribution, sample.shape[2])
    attribution = attribution.squeeze(0).detach().numpy()

    maps.append(attribution)
  return maps

def visualizeMaps(inputs, maps):
  for sample, map1 in zip(inputs,maps):
    # Visualize the sample and the map
    fig, ax = plt.subplots(2, 1, figsize=(5,10))
    ax[0].plot(sample.transpose(1,0))
    ax[0].set_title("sample")
    ax[1].plot(map1.transpose(1,0))
    ax[1].set_title("map")
    plt.tight_layout()
    plt.show()
    print()
