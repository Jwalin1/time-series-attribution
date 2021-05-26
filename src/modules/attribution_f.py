import numpy as np
import matplotlib.pyplot as plt

# for neural network
import torch

# to track progress
from tqdm.notebook import tqdm

# for attribution
# https://captum.ai/api/attribution.html
from captum.attr import Saliency, IntegratedGradients, InputXGradient, GuidedBackprop, LayerGradCam, GuidedGradCam, Lime

# https://github.com/yiskw713/ScoreCAM
from modules.cam import GradCAMpp, SmoothGradCAMpp, ScoreCAM
# https://github.com/yiskw713/RISE
from modules.rise import RISE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def applyMethod(method, model, samples):
  model.eval()
  maps = []
  for sample in tqdm(samples):

    sample = torch.from_numpy(sample).to(device).unsqueeze(0)
    sample = sample.float().requires_grad_(True)
    target = model(sample).argmax(dim=1)[0]
    if method in ["LayerGradCam", "GuidedGradCam"] + ["GradCAMpp", "SmoothGradCAMpp", "ScoreCAM"]:
      interpreter = globals()[method](model, model.features[-3])
    elif method == "RISE":
      interpreter = globals()[method](model, input_size=(sample.shape[2]))
    else:
      interpreter = globals()[method](model)
      
    if method in ["GradCAMpp", "SmoothGradCAMpp", "ScoreCAM"]:
      attribution, idx = interpreter(sample)
    elif method == "RISE":
      attribution = interpreter(sample)[target].view(1,1,-1)
    else:  
      attribution = interpreter.attribute(sample, target=target)

    attribution = attribution.squeeze(0).detach().cpu().numpy()
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