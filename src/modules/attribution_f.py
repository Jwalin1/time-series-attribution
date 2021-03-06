import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from scipy import stats

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
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficent is not defined.")
np.seterr(divide='ignore', invalid='ignore')

captum_methods = ["Saliency", "IntegratedGradients", "InputXGradient", "GuidedBackprop", "LayerGradCam", "GuidedGradCam", "Lime"]
yiskw713_methods = ["GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "RISE"]
approaches = ["replaceWithZero_most", "replaceWithMean_most", "replaceWithInterp_most"]
rand_layers = [0,-1,-2,-3]
percs = [95,90,85,80]
datasets = ["SyntheticAnomaly","CharacterTrajectories","FordA","ElectricDevices","Cricket",
            "LargeKitchenAppliances","PhalangesOutlinesCorrect","NonInvasiveFetalECGThorax1",
            "Wafer","Strawberry","TwoPatterns","Epilepsy","UWaveGestureLibraryAll"]


# function to apply attribution method and return maps
def applyMethod(method, model, inputs):
  # only captum methods support passing multiple inputs at a time
  if method in captum_methods[:-1]: # excluded Lime since it gives a warning when passing multiple inputs
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
  dataLoader = createLoader(inputs, dummyLabels, batch_size=64)

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
    fig, axs = plt.subplots(2, 1, figsize=(5,10))
    axs[0].plot(sample.transpose(1,0))
    axs[0].set_title("sample")
    axs[1].plot(map1.transpose(1,0))
    axs[1].set_title("map")
    axs.tight_layout()
    axs.show()
    print()


# function to replace most/least important points based on percentile
def replace(inputs, maps, n_percentile=90, approach="replaceWithZero_most", reduce_channel=False):
  repl, imp= approach.split('_')
  n_samples, n_channels, sample_lens = inputs.shape
  replaced_samples = []
  for sample, map1 in tqdm(zip(inputs,maps), total=len(inputs), leave=False, desc=repl):

    if reduce_channel == "mean":
      map1 = np.mean(map1,axis=0, keepdims=True)
    elif reduce_channel == "max":
      map1 = np.max(map1,axis=0, keepdims=True)

    nth_percentile = np.percentile(map1,n_percentile)
    if repl == "replaceWithZero": # simply replace the point with 0
      replaceWith = 0
      new_sample = np.where(map1 < nth_percentile, replaceWith, sample) if imp == "least" else np.where(map1 > nth_percentile, replaceWith, sample)
    elif repl == "replaceWithMean": # replace the point with the channel mean
      replaceWith = np.array([np.mean(sample, axis=1),]*sample_lens).transpose()
      new_sample = np.where(map1 < nth_percentile, replaceWith, sample) if imp == "least" else np.where(map1 > nth_percentile, replaceWith, sample)
    elif repl == "replaceWithInterp":
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
            if i==0 and j==len(ch_vals)-1:  vals.extend([0]*(j-i))  # all points to be replaced
            elif i==0:                      vals.extend([ch_vals[j]]*(j-i)) # initial point to be replaced as well
            elif j==len(ch_vals):         vals.extend([ch_vals[i-1]]*(j-i)) # last point to be replaced as well
            else: vals.extend(np.linspace(ch_vals[i-1], ch_vals[j], (j-i))) # replace intermediate point
            i = j
          else:
            vals.append(ch_vals[i])
            i += 1
        new_sample[channel] = vals
    elif repl == "remove":  # remove the points and match the sample to be of the original size
      imp_pts = np.where(map1 < nth_percentile) if imp == "least" else np.where(map1 > nth_percentile)
      new_sample = np.zeros(sample.shape)
      for channel in range(n_channels):
        indxs = imp_pts[1][imp_pts[0]==channel]
        indxs = [i for i in range(sample_lens) if i not in indxs]
        vals = sample[channel][indxs]
        new_sample[channel] = np.interp(np.linspace(0,len(vals)-1,sample_lens),range(len(vals)), vals)

    replaced_samples.append(new_sample)
  return np.array(replaced_samples)


# function to evaluate all param combinations and return a nested dict of accuracies
def gridEval(model, inputs, labels, params):
  if "approaches" in params:  # else the global list of approaches will be used
    approaches1 = params["approaches"]
  else:
    approaches1 = approaches  

  if "methods" not in params:
    methods = captum_methods + yiskw713_methods
  else:
    methods = params["methods"]

  percs = params["percs"]
  rand_layers = params["rand_layers"]
  rand_type = "upto" if "rand_type" not in params else params["rand_type"]


  # compute original attribution maps for comparison with randomized
  maps_original = {}
  for method in tqdm(methods, leave=False, desc="original attribution maps"):
    np.random.seed(0)
    torch.manual_seed(0)
    maps_original[method] = applyMethod(method, model, inputs)

  accs_randModel = {}
  dataLoader = createLoader(inputs, labels, batch_size=500)
  # baseline accuracy
  no_randomized = evaluate(model, dataLoader, output_dict=True)
  accs_randModel["no_randomized_acc"] = no_randomized["accuracy"]
  accs_randModel["no_randomized_macroAvgF1"] = no_randomized["macro avg"]['f1-score']
  accs_randModel["no_randomized_weightedAvgF1"] = no_randomized["weighted avg"]['f1-score']
  for rand_layer in tqdm(rand_layers, leave=False, desc="randomized"):
    # get model with last n layers randomized
    rand_model = randomize_layers(model, rand_layer, rand_type)
    no_replace = evaluate(rand_model, dataLoader, output_dict=True)

    accs_attribMethods = {}
    accs_attribMethods["no_replace_acc"] = no_replace["accuracy"]
    accs_attribMethods["no_replace_macroAvgF1"] = no_replace["macro avg"]['f1-score']
    accs_attribMethods["no_replace_weightedAvgF1"] = no_replace["weighted avg"]['f1-score']

    for method in tqdm(methods, leave=False, desc="methods"):
      accs_replaceApproach = {}
      # correlation between attribution map of original and replaced model
      np.random.seed(0)
      torch.manual_seed(0)

      maps_randomized = applyMethod(method, rand_model, inputs)
      spearmanCorrs, pearsonCorrs = {}, {}
      for method1 in maps_original:
        spearman_corrs, pearson_corrs = [], []
        for map_original,map_randomized in zip(maps_original[method1],maps_randomized):
          if np.isfinite(map_randomized).all() and np.isfinite(map_original).all():
            corr = stats.spearmanr(np.mean(map_original,axis=0), np.mean(map_randomized,axis=0))[0]
            #corr = stats.spearmanr(np.max(map_original,axis=0), np.max(map_randomized,axis=0))[0]
            if ~np.isnan(corr): spearman_corrs.append(corr)
            corr = stats.pearsonr(np.mean(map_original,axis=0), np.mean(map_randomized,axis=0))[0]
            #corr = stats.pearsonr(np.max(map_original,axis=0), np.max(map_randomized,axis=0))[0]
            if ~np.isnan(corr): pearson_corrs.append(corr)
        if spearman_corrs:
          spearmanCorrs[method1] = np.mean(spearman_corrs)
        if pearson_corrs:
          pearsonCorrs[method1] = np.mean(pearson_corrs)
      accs_replaceApproach["spearmanCorr"] = spearmanCorrs
      accs_replaceApproach["pearsonCorr"] = pearsonCorrs

      for approach in tqdm(approaches1, leave=False, desc="approaches"):
        accs = {}
        for perc in tqdm(percs, leave=False, desc="percentile"):
          replacedInputs = replace(inputs, maps_randomized, n_percentile=perc, approach=approach)
          dataLoader1 = createLoader(replacedInputs, labels, batch_size=500)
          perc_dict = evaluate(rand_model, dataLoader1, output_dict=True)
          accs[str(perc)+"_acc"] = perc_dict["accuracy"]
          accs[str(perc)+"_macroAvgF1"] = perc_dict["macro avg"]['f1-score']
          accs[str(perc)+"_weightedAvgF1"] = perc_dict["weighted avg"]['f1-score']
        accs_replaceApproach[approach] = accs
      accs_attribMethods[method] = accs_replaceApproach
    accs_randModel[rand_layer] = accs_attribMethods

  return accs_randModel


# function to visualize the results of evaluation
def visEval(params, accs, savefig):
  params = OrderedDict(params)
  datasets1 = params.pop("datasets", None)
  if datasets1 is None: datasets1 = datasets
  fig, axs = plt.subplots()
  for dataset in tqdm(datasets1, leave=False, desc="datasets"):
    params_list = ["dataset", "rand_layer", "method", "approach", "perc"]
    params["dataset"] = dataset

    # to ensure "dataset" is at the beginning of title
    params.move_to_end("dataset", last=False)
    methods = captum_methods + yiskw713_methods
    plot_paramKeys = []
    plot_title = ""
    # form the title and get the keys to plot
    for param in params:
      if params[param] is None:
        plot_paramKeys.append(param)
      plot_title += param + ':' + str(params[param]) + " "
    plot_title = plot_title[:-1]

    plot_paramKey1, plot_paramKey2 = plot_paramKeys
    if plot_paramKey1 == "method":  plot_params1 = methods
    elif plot_paramKey1 == "approach":  plot_params1 = approaches
    elif plot_paramKey1 == "rand_layer":  plot_params1 = rand_layers
    elif plot_paramKey1 == "perc":  plot_params1 = percs
    if plot_paramKey2 == "method":  plot_params2 = methods
    elif plot_paramKey2 == "approach":  plot_params2 = approaches
    elif plot_paramKey2 == "rand_layer":  plot_params2 = rand_layers
    elif plot_paramKey2 == "perc":  plot_params2 = percs
    
    ys = []
    for i,plot_paramValue1 in enumerate(plot_params1):
      x = []; y = []
      for plot_paramValue2 in plot_params2:
        tmp_dict = accs
        for param in params_list:
          if plot_paramKey1 == param:
            tmp_dict = tmp_dict[str(plot_paramValue1)]
          elif plot_paramKey2 == param:
            tmp_dict = tmp_dict[str(plot_paramValue2)]
          else:
            tmp_dict = tmp_dict[str(params[param])]
          if type(tmp_dict) is dict and "no_randomized_acc" in tmp_dict:
            baseline = tmp_dict["no_randomized_acc"]
        x.append(plot_paramValue2);  y.append(tmp_dict)
      # use bar chart if x axis data is string else line chart
      if isinstance(plot_params2[0], str):
        n_bars = len(plot_params1)
        width = 0.6/(n_bars-1)
        axs.bar(np.arange(len(x))+width*(i-(n_bars-1)/2), y, width, label=plot_paramValue1)
        ys.append(y)
      else:  
        axs.plot(range(len(x)), y, label=plot_paramValue1)
    # elevate bottom ylim to see differences more clearly
    if isinstance(plot_params2[0], str):
      min_y = np.min(ys)
      axs.set_ylim(bottom=min_y-min_y/50)       
    plt.xticks(rotation=90)
    axs.set_xticks(range(len(plot_params2)))
    axs.set_xticklabels(plot_params2)
    bottom, top = axs.get_ylim()
    if top >= baseline*0.8:
      x = [baseline]*len(plot_params2)
      axs.plot(x, label="baseline", ls='--')
      diff = top - bottom
      axs.set_ylim(top=baseline + diff/20)
    axs.set_ylabel("accuracy")
    axs.set_xlabel(plot_paramKey2)
    axs.set_title(plot_title)
    axs.legend(title=plot_paramKey1,loc='center left', bbox_to_anchor= (1.0, 0.5))

    if not savefig:
      plt.show()
    else:
      savefig_dir = "results/plots/%s/" % (dataset)
      if not os.path.exists(savefig_dir):
        os.makedirs(savefig_dir)   # create a dir to store plots
      fig_name = plot_title.replace(" ", "_")
      plt.savefig(savefig_dir + fig_name +".png", bbox_inches='tight')
    plt.cla()
  plt.close(fig)  
  return

def visAttrib(model, inputs, labels, params):
  rand_type = "upto" if "rand_type" not in params else params["rand_type"]

  for input, label in zip(inputs, labels):
    input = np.expand_dims(input, 0)
    label = np.expand_dims(label, 0)
    for rand_layer in tqdm(params["rand_layers"], leave=False, desc="randomized"):
      rand_model = randomize_layers(model, rand_layer, rand_type)
      for method in tqdm(params["methods"], leave=False, desc="methods"):
        map1 = applyMethod(method, rand_model, input)
        for approach in tqdm(params["approaches"], leave=False, desc="approaches"):
          for perc in tqdm(params["percs"], leave=False, desc="percentile"):
            fig, axs = axs.subplots(3)
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
            axs.show()