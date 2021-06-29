import numpy as np
import matplotlib.pyplot as plt

import warnings # to filter out warnings
import torch    # for neural network
from tqdm.auto import tqdm   # to track progress

# for attribution
# https://captum.ai/api/attribution.html
from captum.attr import Saliency, IntegratedGradients, InputXGradient, GuidedBackprop, LayerGradCam, GuidedGradCam, Lime

# https://github.com/yiskw713/ScoreCAM
from modules.cam import GradCAMpp, SmoothGradCAMpp, ScoreCAM
# https://github.com/yiskw713/RISE
from modules.rise import RISE

from modules.network_f import evaluate, randomize_layers
from modules.data_f import createLoader, interpolate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes ")
warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU activations.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def applyMethod(method, model, inputs):
  captum_methods = ["Saliency", "IntegratedGradients", "InputXGradient", "GuidedBackprop", "LayerGradCam", "GuidedGradCam", "Lime"]
  del captum_methods[-1]    # exclude Lime since it gives a warning when passing multiple inputs
  if method in captum_methods:
    maps = applyMethodBatch(method, model, inputs)
  else:
    maps = applyMethodSample(method, model, inputs)

  maps = np.array(maps)
  _, _, sample_len = inputs.shape
  _, _, map_len = maps.shape
  if sample_len != map_len:
      maps = interpolate(maps,sample_len)

  return maps

def applyMethodBatch(method, model, inputs):
  dummyLabels = [0]*len(inputs)
  dataLoader = createLoader(inputs, dummyLabels)

  model.eval()
  maps = []
  for i, (inputBatch,_) in enumerate(tqdm(dataLoader, leave=False, desc="eval")):
    inputBatch = inputBatch.to(device).float().requires_grad_(True)
    targetBatch = model(inputBatch).argmax(dim=1)

    if method in ["LayerGradCam", "GuidedGradCam"] + ["GradCAMpp", "SmoothGradCAMpp", "ScoreCAM"]:
      interpreter = globals()[method](model, model.features[-3])
    elif method == "RISE":
      interpreter = globals()[method](model, input_size=(inputBatch.shape[2]))
    else:
      interpreter = globals()[method](model)

    if method in ["GradCAMpp", "SmoothGradCAMpp", "ScoreCAM"]:
      attribution, idx = interpreter(inputBatch)
    elif method == "RISE":
      attribution = interpreter(inputBatch)[targetBatch].view(1,1,-1)
    else:
      attribution = interpreter.attribute(inputBatch, target=targetBatch)

    attribution = attribution.detach().cpu().numpy()
    maps.extend(attribution)
  return maps

def applyMethodSample(method, model, samples):
  model.eval()
  maps = []
  for sample in tqdm(samples, leave=False, desc=method):

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


def replace(inputs, maps, n_percentile=90, imp="most", approach="replaceWithZero"):
  n_samples, n_channels, sample_lens = inputs.shape
  replaced_samples = []
  for sample, map1 in tqdm(zip(inputs,maps), total=len(inputs), leave=False, desc=approach):
    nth_percentile = np.percentile(map1,n_percentile)
    if approach == "replaceWithZero":
      replaceWith = 0
      new_sample = np.where(map1 < nth_percentile, replaceWith, sample) if imp == "least" else np.where(map1 > nth_percentile, replaceWith, sample)
    elif approach == "replaceWithMean":
      replaceWith = np.array([np.mean(sample, axis=1),]*sample_lens).transpose()
      new_sample = np.where(map1 < nth_percentile, replaceWith, sample) if imp == "least" else np.where(map1 > nth_percentile, replaceWith, sample)
    elif approach == "replaceWithInterp":
      nth_percentile = np.percentile(map1,n_percentile)
      imp_pts = np.where(map1 < nth_percentile) if imp == "least" else np.where(map1 > nth_percentile)
      new_sample = np.zeros(sample.shape)
      for channel in range(n_channels):
        indxs = imp_pts[1][imp_pts[0]==channel]
        #indxs = [i for i in range(sample_lens) if i not in indxs]
        vals = []
        ch_vals = sample[channel]
        i=0
        while i in range(len(ch_vals)):
          if i in indxs:
            j=i+1
            while j in indxs: j += 1
            if i==0 and j==len(ch_vals)-1:  vals.extend([0]*(j-i))
            elif i==0:                      vals.extend([ch_vals[j]]*(j-i))
            elif j==len(ch_vals):         vals.extend([ch_vals[i-1]]*(j-i))
            else: vals.extend(np.linspace(ch_vals[i-1], ch_vals[j], (j-i)))
            i = j
          else:
            vals.append(ch_vals[i])
            i += 1
        new_sample[channel] = vals
    elif approach == "remove":
      imp_pts = np.where(map1 < nth_percentile) if imp == "least" else np.where(map1 > nth_percentile)
      new_sample = np.zeros(sample.shape)
      for channel in range(n_channels):
        indxs = imp_pts[1][imp_pts[0]==channel]
        indxs = [i for i in range(sample_lens) if i not in indxs]
        vals = sample[channel][indxs]
        new_sample[channel] = np.interp(np.linspace(0,len(vals)-1,sample_lens),range(len(vals)), vals)

    replaced_samples.append(new_sample)
  return np.array(replaced_samples)


def gridEval(model, inputs, labels, params):
  if "approaches" not in params:
    approaches = ["replaceWithZero", "replaceWithMean", "replaceWithInterp"]
  else:
    approaches = params["approaches"]

  captum_methods = ["Saliency", "IntegratedGradients", "InputXGradient", "GuidedBackprop", "LayerGradCam", "GuidedGradCam", "Lime"]
  yiskw713_methods = ["GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "RISE"]
  if "methods" not in params:
    methods = captum_methods + yiskw713_methods
  else:
    methods = params["methods"]

  percs = params["percs"]
  rand_layers = params["rand_layers"]


  accs_randModel = {}
  for rand_layer in tqdm(rand_layers, leave=False, desc="randomized"):
    # get model with last n layers randomized
    rand_model = randomize_layers(model, rand_layer)

    accs_attribMethods = {}
    dataLoader = createLoader(inputs, labels)
    accs_attribMethods["no_replace"] = evaluate(rand_model, dataLoader, output_dict=True)["accuracy"]

    for method in tqdm(methods, leave=False, desc="methods"):
      accs_replaceApproach = {}
      maps = applyMethod(method, rand_model, inputs)
      for approach in tqdm(approaches, leave=False, desc="approaches"):
        accs = {}
        for perc in tqdm(percs, leave=False, desc="percentile"):
          replacedInputs = replace(inputs, maps, n_percentile=perc, approach=approach)
          dataLoader = createLoader(replacedInputs, labels)
          accs[perc] = evaluate(rand_model, dataLoader, output_dict=True)["accuracy"]
        accs_replaceApproach[approach] = accs
      accs_attribMethods[method] = accs_replaceApproach
    accs_randModel[rand_layer] = accs_attribMethods

  return accs_randModel


def visEval(params, accs):
  for dataset in params["datasets"]:
    for rand_layer in params["rand_layers"]:
      for method in params["methods"]:
        for approach in params["approaches"]:
          print("dataset:%s, rand_layer:%s, method:%s, approach:%s" % (dataset,rand_layer,method,approach))
          x, y = zip(*accs[dataset][rand_layer][method][approach].items())
          # append initial accuracy without replacement
          x = list(x); x.insert(0,"100")
          y = list(y); y.insert(0,accs[dataset][rand_layer]["no_replace"])
          plt.plot(x,y)
          plt.show()

def visAttrib(model, inputs, labels, params):
  for input, label in zip(inputs, labels):
    input = np.expand_dims(input, 0)
    label = np.expand_dims(label, 0)    
    for rand_layer in tqdm(params["rand_layers"], leave=False, desc="randomized"):
      rand_model = randomize_layers(model, rand_layer)
      for method in tqdm(params["methods"], leave=False, desc="methods"):
        map1 = applyMethod(method, rand_model, input)
        for approach in tqdm(params["approaches"], leave=False, desc="approaches"):
          for perc in tqdm(params["percs"], leave=False, desc="percentile"):
            fig, axs = plt.subplots(3)
            print("rand_layer:%s, method:%s, approach:%s" % (rand_layer,method,approach))
            replacedInput = replace(input, map1, n_percentile=perc, approach=approach)
            dataLoader = createLoader(input, label)
            pred_label = evaluate(rand_model, dataLoader, output_pred=True)[0].cpu().numpy()
            dataLoader = createLoader(replacedInput, label)
            pred_label1 = evaluate(rand_model, dataLoader, output_pred=True)[0].cpu().numpy()
            print("correct label:%s, pred label:%s, pred label after replace:%s" % (label[0],pred_label,pred_label1))
            axs[0].plot(input[0].transpose())
            axs[1].plot(map1[0].transpose())
            axs[2].plot(replacedInput[0].transpose())
            plt.show()