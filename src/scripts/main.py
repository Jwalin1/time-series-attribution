import numpy as np
import argparse

# import modules
from modules import data_f, network_f, network_architectures, attribution_f

# reload module
import importlib
importlib.reload(attribution_f)




def main(args):
  train_inputs, train_labels, test_inputs, test_labels = data_f.getRead_data(args.dataset)
  print("train_inputs shape:",train_inputs.shape)
  dataloaders = data_f.createLoaders(train_inputs, train_labels, test_inputs, test_labels, 32)
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
  # network_f.evaluate(model, dataloaders["test"])

  # x = train_inputs[np.where(train_labels==1)]
  # map1 = attribution_f.applyMethod(args.method, model, [x[0]])
  # attribution_f.visualizeMaps([x[0]], map1)

  selectedInputs = data_f.selectInputs(train_inputs, train_labels, args.n_samples)
  maps = attribution_f.applyMethod(args.method, model, selectedInputs)
  data_f.saveMaps(maps, args.method, args.dataset)
  maps = data_f.loadMaps(args.method, args.dataset)
  attribution_f.visualizeMaps(selectedInputs, maps)
  replacedInputs = attribution_f.replace(selectedInputs, maps, approach="replaceWithZero")



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
