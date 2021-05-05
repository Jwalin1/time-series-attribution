import argparse

# import modules
from modules import data_f, network_f, network_architectures

# reload module
import importlib
importlib.reload(data_f)




def main(args):
  train_inputs, train_labels, test_inputs, test_labels = data_f.getRead_data(args.dataset)
  dataloaders = data_f.createLoaders(train_inputs, train_labels, test_inputs, test_labels, 32)
  trainValLoaders = {"train":dataloaders["train"], "val":dataloaders["val"]}
  num_classes = {"CharacterTrajectories":20,"SyntheticAnomaly":2,"FordA":2,"ElectricDevices":7}
  inp_channels = {"CharacterTrajectories":3,"SyntheticAnomaly":3,"FordA":1,"ElectricDevices":1}

  n, c = num_classes[args.dataset], inp_channels[args.dataset]
  model,criterion,optimizer,scheduler = network_f.setupModel(network_architectures.AlexNet(n, c))
  best_params, last_params = network_f.train_model(model,criterion,optimizer,scheduler,trainValLoaders,epochs=args.epochs)
  model.load_state_dict(best_params)
  network_f.evaluate(model, dataloaders["test"])




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str ,choices=["CharacterTrajectories","SyntheticAnomaly","FordA","ElectricDevices"])
  parser.add_argument("--epochs", type=int)

  args = parser.parse_args()
  main(args)