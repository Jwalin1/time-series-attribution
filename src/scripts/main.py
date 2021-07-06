import numpy as np
import argparse
from tqdm.auto import tqdm

# to process dicts
import json
import ast

# to be able to import other python files
import sys
sys.path.append("../")

# import modules
from modules import data_f, network_f, network_architectures, attribution_f

# reload module
import importlib
importlib.reload(attribution_f)
importlib.reload(data_f)
importlib.reload(network_f)

# change directory to project directory
import os
os.chdir("../../")



def main(args):
  if args.visEvalParams is None:
    if args.gridEvalParams is not None:
      if "datasets" not in args.gridEvalParams:
        datasets = ["SyntheticAnomaly","CharacterTrajectories","FordA","ElectricDevices","Cricket",
                    "LargeKitchenAppliances","PhalangesOutlinesCorrect","NonInvasiveFetalECGThorax1",
                    "Wafer","Strawberry","TwoPatterns","Epilepsy","UWaveGestureLibraryAll"]
      else:
        datasets = args.gridEvalParams["datasets"]
      gridEval_results = {}  
    elif args.visAttribParams is not None:
      datasets = args.visAttribParams["datasets"]    
    else:
      datasets = [args.dataset]

    for dataset in tqdm(datasets, leave=False, desc="datasets"):
      train_inputs, train_labels, test_inputs, test_labels = data_f.getRead_data(dataset)
      print("train_inputs shape:",train_inputs.shape)
      dataloaders = data_f.createLoaders(train_inputs, train_labels, test_inputs, test_labels)
      trainValLoaders = {"train":dataloaders["train"], "val":dataloaders["val"]}
      n_samples, n_channels, sample_lens = train_inputs.shape
      classes = np.unique(train_labels)
      n_classes = len(classes)

      model,criterion,optimizer,scheduler = network_f.setupModel(network_architectures.AlexNet(n_classes, n_channels))
      if args.load is not None:
        network_f.load_state_dict(model, dataset)
      else:
        best_params, last_params = network_f.train_model(model, criterion, optimizer, scheduler,
                                                        trainValLoaders, epochs=args.epochs, earlyStopping=True)
        model.load_state_dict(best_params)
        if args.save is not None:
          network_f.save_state_dict(model, dataset)

      if args.visAttribParams is not None:
        selectedInputs, selectedLabels = data_f.subsample2(train_inputs, train_labels, args.n_samples)
      else:  
        selectedInputs, selectedLabels = data_f.subsample(train_inputs, train_labels, args.n_samples)
      if args.gridEvalParams is not None:
        accs_dict = attribution_f.gridEval(model, selectedInputs, selectedLabels, args.gridEvalParams)
        with open("results/randomization_results/%s.json"%(dataset),"w") as f:
          json.dump(accs_dict,f)
        gridEval_results[dataset] = accs_dict
      elif args.visAttribParams is not None:
        print("dataset:%s"%(dataset))
        attribution_f.visAttrib(model, selectedInputs, selectedLabels, args.visAttribParams)

  else:
    with open("results/randomization_results.json", 'r') as myfile:
      data=myfile.read()
    accs = json.loads(data)
    attribution_f.visEval(args.visEvalParams, accs, args.save)
  if args.gridEvalParams is not None:
    with open("results/randomization_results.json","w") as f:
      json.dump(gridEval_results,f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, help='["SyntheticAnomaly","CharacterTrajectories","FordA","ElectricDevices"]')
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--save", type=str)
  parser.add_argument("--load", type=str)
  parser.add_argument("--n_samples", type=int)
  parser.add_argument("--method", type=str, help="attribution method to be applied")
  parser.add_argument("--gridEvalParams", type=ast.literal_eval, help="dict containing ranges for grid eval params")
  parser.add_argument("--visEvalParams", type=ast.literal_eval, help="dict containing ranges for vis eval params")
  parser.add_argument("--visAttribParams", type=ast.literal_eval, help="dict containing ranges for vis attrib params")
  # keys are {"methods" : ["Saliency","GradCAMpp","SmoothGradCAMpp"],  "approaches" : ["replaceWithMean", "replaceWithInterp"],  "percs" : [99,98,96,92], "rand_layers":-3'}

  args = parser.parse_args()
  main(args)
