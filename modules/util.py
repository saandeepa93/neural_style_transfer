import matplotlib.pyplot as plt
import cv2
import yaml
import torch
from torch import nn
import numpy as np
from sys import exit as e

def imshow(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def show(img):
  plt.imshow(img)
  plt.show()


class MeanShift(nn.Conv2d):
  def __init__(self, gpu_id):
    super(MeanShift, self).__init__(3, 3, kernel_size=1)
    rgb_range=1
    rgb_mean=(0.4488, 0.4371, 0.4040)
    rgb_std=(1.0, 1.0, 1.0)
    sign=-1
    std = torch.Tensor(rgb_std).to(gpu_id)
    self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(gpu_id) / std.view(3, 1, 1, 1)
    self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to(gpu_id) / std
    for p in self.parameters():
        p.requires_grad = False

def numpy2tensor(np_array, gpu_id):
  gpu_id = "cpu"
  if len(np_array.shape) == 2:
      tensor = torch.from_numpy(np_array).unsqueeze(0).float().to(gpu_id)
  else:
      tensor = torch.from_numpy(np_array).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
  return tensor



def get_config(config_path):
  with open(config_path) as file:
    configs = yaml.load(file, Loader = yaml.FullLoader)
  return configs


def laplacian_filter_tensor(img_tensor, gpu_id='cpu'):
  laplacian_filter = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
  laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
  laplacian_conv.weight = nn.Parameter(torch.from_numpy(laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(gpu_id))

  for param in laplacian_conv.parameters():
      param.requires_grad = False

  red_img_tensor = img_tensor[:,0,:,:].unsqueeze(1)
  green_img_tensor = img_tensor[:,1,:,:].unsqueeze(1)
  blue_img_tensor = img_tensor[:,2,:,:].unsqueeze(1)

  red_gradient_tensor = laplacian_conv(red_img_tensor).squeeze(1)
  green_gradient_tensor = laplacian_conv(green_img_tensor).squeeze(1)
  blue_gradient_tensor = laplacian_conv(blue_img_tensor).squeeze(1)
  return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor


def compute_gt_gradient(source_img_tensor, target_img_tensor, mask, gpu_id="cpu"):
    # compute source image gradient
    _, _, sr, sc = source_img_tensor.size()
    _, _, tr, tc = target_img_tensor.size()
    mr, mc = mask.shape
    x_start, y_start = sr//2, sc//2

    red_source_gradient_tensor, green_source_gradient_tensor, blue_source_gradient_tenosr = laplacian_filter_tensor(source_img_tensor, gpu_id)
    red_source_gradient = red_source_gradient_tensor.cpu().data.numpy()[0]
    green_source_gradient = green_source_gradient_tensor.cpu().data.numpy()[0]
    blue_source_gradient = blue_source_gradient_tenosr.cpu().data.numpy()[0]

    red_target_gradient_tensor, green_target_gradient_tensor, blue_target_gradient_tenosr = laplacian_filter_tensor(target_img_tensor, gpu_id)
    red_target_gradient = red_target_gradient_tensor.cpu().data.numpy()[0]
    green_target_gradient = green_target_gradient_tensor.cpu().data.numpy()[0]
    blue_target_gradient = blue_target_gradient_tenosr.cpu().data.numpy()[0]

    canvas_mask = np.zeros((tr, tc))
    canvas_mask[int(x_start-sr*0.5):int(x_start+sr*0.5), int(y_start-sc*0.5):int(y_start+sc*0.5)] = mask

    red_source_gradient = red_source_gradient * mask
    green_source_gradient = green_source_gradient * mask
    blue_source_gradient = blue_source_gradient * mask
    red_foreground_gradient = np.zeros((canvas_mask.shape))
    red_foreground_gradient[int(x_start-sr*0.5):int(x_start+sr*0.5), int(y_start-sc*0.5):int(y_start+sc*0.5)] = red_source_gradient
    green_foreground_gradient = np.zeros((canvas_mask.shape))
    green_foreground_gradient[int(x_start-sr*0.5):int(x_start+sr*0.5), int(y_start-sc*0.5):int(y_start+sc*0.5)] = green_source_gradient
    blue_foreground_gradient = np.zeros((canvas_mask.shape))
    blue_foreground_gradient[int(x_start-sr*0.5):int(x_start+sr*0.5), int(y_start-sc*0.5):int(y_start+sc*0.5)] = blue_source_gradient

    # background gradient
    red_background_gradient = red_target_gradient * (canvas_mask - 1) * (-1)
    green_background_gradient = green_target_gradient * (canvas_mask - 1) * (-1)
    blue_background_gradient = blue_target_gradient * (canvas_mask - 1) * (-1)

    # add up foreground and background gradient
    gt_red_gradient = red_foreground_gradient + red_background_gradient
    gt_green_gradient = green_foreground_gradient + green_background_gradient
    gt_blue_gradient = blue_foreground_gradient + blue_background_gradient

    gt_red_gradient = numpy2tensor(gt_red_gradient, gpu_id)
    gt_green_gradient = numpy2tensor(gt_green_gradient, gpu_id)
    gt_blue_gradient = numpy2tensor(gt_blue_gradient, gpu_id)

    gt_gradient = [gt_red_gradient, gt_green_gradient, gt_blue_gradient]
    return gt_gradient


