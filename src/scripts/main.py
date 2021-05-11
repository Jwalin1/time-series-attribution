import numpy as np
import argparse

# import modules
from modules import data_f, network_f, network_architectures

# reload module
import importlib
importlib.reload(network_f)




def main(args):
  train_inputs, train_labels, test_inputs, test_labels = data_f.getRead_data(args.dataset)
  print("train_inputs shape:",train_inputs.shape)
  dataloaders = data_f.createLoaders(train_inputs, train_labels, test_inputs, test_labels, 32)
  trainValLoaders = {"train":dataloaders["train"], "val":dataloaders["val"]}
  n_samples, n_channels, sample_lens = train_inputs.shape
  n_classes = len(np.unique(train_labels))

  model,criterion,optimizer,scheduler = network_f.setupModel(network_architectures.AlexNet(n_classes, n_channels))
  best_params, last_params = network_f.train_model(model, criterion, optimizer, scheduler,
                                                    trainValLoaders, epochs=args.epochs, earlyStopping=True)
  model.load_state_dict(best_params)
  network_f.evaluate(model, dataloaders["test"])




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str ,help='["CharacterTrajectories","SyntheticAnomaly","FordA","ElectricDevices"]')
  parser.add_argument("--epochs", type=int)

  args = parser.parse_args()
  main(args)