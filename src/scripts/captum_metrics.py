# used to evaluate and print captum metrics


import json
import torch
import captum
import numpy as np
from tqdm.auto import tqdm
from captum.metrics import infidelity, sensitivity_max

# to be able to import other python files
import sys
sys.path.append("../")

from modules import data_f, network_f, network_architectures, attribution_f

import warnings
warnings.filterwarnings("ignore")

# change directory to project directory
import os
os.chdir("../../")

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


captum_methods = ["Saliency", "IntegratedGradients", "InputXGradient", "GuidedBackprop", "GuidedGradCam", "Lime"]
methods = captum_methods

datasets = ["SyntheticAnomaly","CharacterTrajectories","FordA","ElectricDevices","Cricket",
            "LargeKitchenAppliances","PhalangesOutlinesCorrect","NonInvasiveFetalECGThorax1",
            "Wafer","Strawberry","TwoPatterns","Epilepsy","UWaveGestureLibraryAll"]


def perturb_fn(inputs):
  noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).to(device).float()
  return noise, inputs - noise

infids = {}
for dataset in tqdm(datasets, leave=False, desc="datasets"):
  method_infid = {}
  train_inputs, train_labels, test_inputs, test_labels = data_f.getRead_data(dataset)
  selectedInputs, selectedLabels = data_f.subsample(train_inputs, train_labels, 1000)
  n_samples, n_channels, sample_lens = train_inputs.shape
  classes = np.unique(train_labels)
  n_classes = len(classes)

  model,criterion,optimizer,scheduler = network_f.setupModel(network_architectures.AlexNet(n_classes, n_channels))
  network_f.load_state_dict(model, dataset)
  
  for method in tqdm(methods, leave=False, desc="methods"):    
    maps = attribution_f.applyMethod(method, model, selectedInputs)
    infid = np.mean([captum.metrics.infidelity(forward_func=model, perturb_func=perturb_fn, inputs=torch.tensor(selectedInputs).to(device).float(), attributions=torch.tensor(maps).to(device).float(), target=int(class_)).cpu().numpy() for class_ in classes])
    method_infid[method] = str(infid)
  infids[dataset] = method_infid


with open('results/captum_metrics.json', 'w') as fp:
    json.dump(infids, fp) 