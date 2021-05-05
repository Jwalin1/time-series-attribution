import os
import numpy as np

# to read data
import requests
from scipy.io import loadmat, arff
import pickle
import zipfile

# to create dataset and dataloaders
from torch.utils.data import DataLoader, Dataset

# for splitting data
from sklearn.model_selection import train_test_split
np.random.seed(0)


# download file from url
def download_file(url,saveAs):
  print("downloading file ",url)
  if not os.path.exists(saveAs):
    r = requests.get(url, allow_redirects=True)
    open(saveAs, 'wb').write(r.content)
    print("file downloaded")
  else:
    print("file already exists")

# download file from url
def extract_zip(path_to_zip_file, directory_to_extract_to):
  if not os.path.exists(directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
      zip_ref.extractall(directory_to_extract_to)
    print("folder extracted from zip")
  else:
    print("zip already extracted")


def getRead_data(dataset):

  # get data
  if dataset == "CharacterTrajectories":
    fsource = "http://www.timeseriesclassification.com/Downloads/CharacterTrajectories.zip"
    fname = fsource[fsource.rindex('/')+1:] # fname = "CharacterTrajectories.zip"
    download_file(url = fsource, saveAs = fname)
    extract_zip(fname, "CharacterTrajectories")

    data, meta = arff.loadarff(open("CharacterTrajectories/CharacterTrajectories_TRAIN.arff"))
    train_inputs, train_labels = zip(*data)
    data, meta = arff.loadarff(open("CharacterTrajectories/CharacterTrajectories_TEST.arff"))
    test_inputs, test_labels = zip(*data)

    # convert to np array
    train_inputs = np.array([ [ np.array(list(channel), dtype=float) for channel in input ]  for input in train_inputs])
    test_inputs = np.array([ [ np.array(list(channel), dtype=float) for channel in input ]  for input in test_inputs])
    train_inputs, train_labels = np.nan_to_num(train_inputs), np.array(train_labels, dtype=int)
    test_inputs, test_labels = np.nan_to_num(test_inputs), np.array(test_labels, dtype=int)
    train_labels -= 1; test_labels -= 1 # change labels from [1,20] to [0,19]
    
  elif dataset == "SyntheticAnomaly":
    download_file(url = "https://drive.google.com/u/0/uc?id=1CdYxeX8g9wxzSnz6R51ELmJJuuZ3xlqa&export=download",
                          saveAs = "anomaly_dataset.pickle")

    infile = open("anomaly_dataset.pickle",'rb')
    data = pickle.load(infile)
    infile.close()

    # read data
    train_inputs, train_labels, test_inputs, test_labels = data

    train_inputs = np.transpose(train_inputs, (0,2,1))
    test_inputs = np.transpose(test_inputs, (0,2,1))
    # sample_len = 50;  classes = ["normal","anomaly"]

  elif dataset == "FordA":
    fsource = "http://www.timeseriesclassification.com/Downloads/FordA.zip"
    fname = fsource[fsource.rindex('/')+1:] # fname = "FordA.zip"
    download_file(url = fsource, saveAs = fname)
    extract_zip(fname, "FordA")

    data, meta = arff.loadarff(open("FordA/FordA_TRAIN.arff"))
    train_inputs, train_labels = zip(*[(list(sample)[:-1],list(sample)[-1]) for sample in data])
    data, meta = arff.loadarff(open("FordA/FordA_TEST.arff"))
    test_inputs, test_labels = zip(*[(list(sample)[:-1],list(sample)[-1]) for sample in data])
    train_inputs, train_labels = np.expand_dims(np.array(train_inputs),1), np.array(train_labels, dtype="int")
    test_inputs, test_labels = np.expand_dims(np.array(test_inputs),1), np.array(test_labels, dtype="int")
    train_labels, test_labels = (train_labels+1)/2, (test_labels+1)/2 # change labels from [-1,1] to [0,1]

  elif dataset == "ElectricDevices":
    fsource = "http://www.timeseriesclassification.com/Downloads/ElectricDevices.zip"
    fname = fsource[fsource.rindex('/')+1:] # fname = "ElectricDevices.zip"
    download_file(url = fsource, saveAs = fname)
    extract_zip(fname, "ElectricDevices") 

    data, meta = arff.loadarff(open("ElectricDevices/ElectricDevices_TRAIN.arff"))
    train_inputs, train_labels = zip(*[(list(sample)[:-1],list(sample)[-1]) for sample in data])
    data, meta = arff.loadarff(open("ElectricDevices/ElectricDevices_TEST.arff"))
    test_inputs, test_labels = zip(*[(list(sample)[:-1],list(sample)[-1]) for sample in data])
    train_inputs, train_labels = np.expand_dims(np.array(train_inputs),1), np.array(train_labels, dtype="int")
    test_inputs, test_labels = np.expand_dims(np.array(test_inputs),1), np.array(test_labels, dtype="int")
    train_labels -= 1; test_labels -= 1 # change labels from [1,7] to [0,6]

  return train_inputs, train_labels, test_inputs, test_labels      

# create dataset and dataloaders
class mydataset(Dataset):
  def __init__(self, inputs, labels):
    self.inputs = inputs
    self.labels = labels

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    input = self.inputs[index]
    label = self.labels[index]
    return input,label

# function to create train, val and test loaders
def createLoaders(train_inputs, train_labels, test_inputs, test_labels, batch_size, val_percent=.25):
  train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=val_percent, random_state=0)

  train_dataset = mydataset(train_inputs, train_labels)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  val_dataset = mydataset(val_inputs, val_labels)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  test_dataset = mydataset(test_inputs, test_labels)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  dataloaders = {"train":train_loader, "val":val_loader, "test":test_loader}
  return dataloaders