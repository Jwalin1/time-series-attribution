import numpy as np
import matplotlib.pyplot as plt

# for neural network
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_grad(model, sample):
  sample = torch.from_numpy(sample).to(device).float()
  sample = sample.unsqueeze(0)
  _ = sample.requires_grad_()

  model.eval()
  # Retrieve output from the image
  output = model(sample)

  # get the logits
  pred = model(sample)
  pred_class = pred.argmax(dim=1)[0]
  #print("prediction:",pred_class)
  
  # grad wrt to predicted class
  pred[:, pred_class].backward()
  return sample

def applyMethod(method, model, samples):
  maps = []
  for sample in samples:
    map1 = globals()[method](model, sample)
    maps.append(map1)
  return maps  

def salMap(model, sample):
  sample = compute_grad(model, sample)

  # In this case, we look at dim=1. Recall the shape (batch_size, channel, time_stamp)
  saliency = sample.grad.data.abs().squeeze().cpu()
  # adjust shape to [sample_len, n_channels]
  saliency = saliency.transpose(0,1) if len(saliency.shape) > 1 else saliency.reshape(-1,1)
  return saliency

def visualizeMaps(inputs, maps):
  for sample, map1 in zip(inputs,maps):
    # Visualize the sample and the map
    fig, ax = plt.subplots(2, 1, figsize=(5,10))
    ax[0].plot(sample.transpose(1,0))
    ax[0].set_title("sample")
    #ax[0].axis('off')
    ax[1].plot(map1)
    ax[1].set_title("map")
    #ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    print()

def gradCAM(model, sample):
  sample = compute_grad(model, sample)

  # pull the gradients out of the model
  gradients = model.get_activations_gradient().cpu().detach()

  # pool the gradients across the channels
  pooled_gradients = torch.mean(gradients, dim=[0, 2])

  # get the activations of the last convolutional layer
  activations = model.get_activations(sample).cpu().detach()

  # weight the channels by corresponding gradients
  for i in range(256):
    activations[:, i, :] *= pooled_gradients[i]
      
  # average the channels of the activations
  heatmap = torch.mean(activations, dim=1).squeeze()

  # relu on top of the heatmap
  # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
  heatmap = np.maximum(heatmap, 0)

  # normalize the heatmap
  heatmap /= torch.max(heatmap)
  return heatmap

  # # Visualize the image and the saliency map
  # fig, ax = plt.subplots(3, 1, figsize=(5,10))
  # ax[0].plot(sample.squeeze(0).transpose(0,1).cpu().detach().numpy())
  # ax[0].set_title("sample")
  # #ax[0].axis('off')

  # # draw the heatmap
  # ax[1].plot(heatmap)
  # ax[1].set_title("grad cam")
  # #ax[1].axis('off')

  # # resize to plot superimposed series
  # n, ch, len1 = sample.shape
  # len2 = len(heatmap)
  # x = np.linspace(0,len2-1,len1)
  # xp = list(range(len2))
  # heatmap = np.interp(x,xp,heatmap)
  # ax[2].plot(heatmap, alpha=0.5)
  # ax[2].plot(sample.squeeze(0).transpose(0,1).cpu().detach().numpy())
  # ax[2].set_title("superimposed")
  # #ax[2].axis('off')
  # plt.tight_layout()
  # plt.show()  