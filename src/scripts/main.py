import numpy as np
import argparse

import sys
sys.path.append("../")

# import modules
from modules import data_f, network_f, network_architectures, attribution_f

# reload module
import importlib
importlib.reload(attribution_f)

import os
os.chdir("../")



def main(args):
  train_inputs, train_labels, test_inputs, test_labels = data_f.getRead_data(args.dataset)
  print("train_inputs shape:",train_inputs.shape)
  dataloaders = data_f.createLoaders(train_inputs, train_labels, test_inputs, test_labels)
  trainValLoaders = {"train":dataloaders["train"], "val":dataloaders["val"]}
  n_samples, n_channels, sample_lens = train_inputs.shape
  classes = np.unique(train_labels)
  n_classes = len(classes)

  model,criterion,optimizer,scheduler = network_f.setupModel(network_architectures.AlexNet(n_classes, n_channels))
  if args.load is not None:
    network_f.load_state_dict(model, args.dataset)
  else:
    best_params, last_params = network_f.train_model(model, criterion, optimizer, scheduler,
                                                      trainValLoaders, epochs=args.epochs, earlyStopping=True)
    model.load_state_dict(best_params)
    if args.save is not None:
      network_f.save_state_dict(model, args.dataset)
  # print(network_f.evaluate(model, dataloaders["test"]))

  # x = train_inputs[np.where(train_labels==1)][:2]
  # maps = attribution_f.applyMethod(args.method, model, x)
  # attribution_f.visualizeMaps(x, maps)
  # dataloader = data_f.createLoader(x, [1,1])
  # print(network_f.evaluate(model, dataloader))
  # replacedInputs = attribution_f.replace(x, maps, n_percentile=90, approach="replaceWithInterp")
  # attribution_f.visualizeMaps(x, replacedInputs)
  # dataloader = data_f.createLoader(replacedInputs, [1,1])
  # print(network_f.evaluate(model, dataloader))

  selectedInputs, selectedLabels = data_f.subsample(train_inputs, train_labels, args.n_samples)
  maps = attribution_f.applyMethod(args.method, model, selectedInputs)
  attribution_f.visualizeMaps(selectedInputs, maps)
  data_f.saveMaps(maps, args.method, args.dataset)
  maps = data_f.loadMaps(args.method, args.dataset)

  #print(attribution_f.gridEval(model, selectedInputs, selectedLabels, maps))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, help='["CharacterTrajectories","SyntheticAnomaly","FordA","ElectricDevices"]')
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--save", type=str)
  parser.add_argument("--load", type=str)
  parser.add_argument("--n_samples", type=int)
  parser.add_argument("--method", type=str, help="attribution method to be applied")

  args = parser.parse_args()
  main(args)
