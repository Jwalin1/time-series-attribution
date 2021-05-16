import os
import numpy as np

# to read data
import requests
from scipy.io import loadmat, arff
import pickle
import zipfile

# to track progress
from tqdm.notebook import tqdm

# to create dataset and dataloaders
from torch.utils.data import DataLoader, Dataset

# for splitting, standardizing and normalizing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

def breakData(data):
  inputs = []
  labels = []
  for sample in tqdm(data, leave=False):

    if isinstance(sample[0], np.ndarray):   # check if input has more than 1 channel
      channels = [list(channel) for channel in sample[0]]  
      inputs.append(channels)  
    else:
      inputs.append([list(sample)[:-1]])

    labels.append(sample[-1])

  inputs = np.nan_to_num(inputs)
  try:  labels = np.array(labels, dtype=float)
  except Exception: print("non numeric labels")

  return inputs, labels

def download_timeSeries(dataset):
  fsource = "http://www.timeseriesclassification.com/Downloads/" + dataset + ".zip"
  fname = dataset + ".zip"
  download_file(url = fsource, saveAs = fname)
  extract_zip(fname, dataset)

  data, meta = arff.loadarff("%s/%s_TRAIN.arff" % (dataset,dataset))
  train_inputs, train_labels = breakData(data)
  data, meta = arff.loadarff("%s/%s_TEST.arff" % (dataset,dataset))
  test_inputs, test_labels = breakData(data)

  # adjust labels to begin from 0
  oldLabels = list(np.unique(train_labels))
  newLabels = np.arange(len(oldLabels))
  # elementwise comparison fails when using np.where on byte'string' data so used list instead
  train_labels = np.array([ newLabels[oldLabels.index(label)] for label in train_labels])
  test_labels = np.array([ newLabels[oldLabels.index(label)] for label in test_labels])

  return train_inputs, train_labels, test_inputs, test_labels


def getRead_data(dataset):
  curr_dir = os.getcwd()
  if not os.path.exists("datasets"):
    os.mkdir("datasets"); os.chdir("datasets")    # create a dir to store datasets
  else:
    os.chdir("datasets")
  # get data
  if dataset == "SyntheticAnomaly":
    download_file(url = "https://drive.google.com/u/0/uc?id=1CdYxeX8g9wxzSnz6R51ELmJJuuZ3xlqa&export=download",
                          saveAs = "anomaly_dataset.pickle")

    infile = open("anomaly_dataset.pickle",'rb')
    data = pickle.load(infile)
    infile.close()

    # read data
    train_inputs, train_labels, test_inputs, test_labels = data
    train_inputs = np.transpose(train_inputs, (0,2,1))
    test_inputs = np.transpose(test_inputs, (0,2,1))

  else:
    train_inputs, train_labels, test_inputs, test_labels = download_timeSeries(dataset)

    if dataset == "UWaveGestureLibraryAll":
    # change shape from n,1,945 to n,3,315 as it is originally supposed to have 3 channels which can also be seen by plotting
      train_inputs, test_inputs = train_inputs.reshape(-1,3,315), test_inputs.reshape(-1,3,315)

  os.chdir(curr_dir)
  train_inputs, test_inputs = standard_normal(train_inputs), standard_normal(test_inputs)
  return train_inputs, train_labels, test_inputs, test_labels


def standard_normal(inputs):
  original_shape = inputs.shape
  new_inputs = []
  for sample in tqdm(inputs, leave=False):
    channels = []
    for channel in sample:
      channel = channel.reshape(-1,1)
      channel = StandardScaler().fit_transform(channel)
      channel = MinMaxScaler(feature_range=(-1,1)).fit_transform(channel)
      channels.append(channel)
    new_inputs.append(channels)
  return np.reshape(new_inputs, original_shape) 


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