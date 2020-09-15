import torch
from torch import nn, optim
import skimage.io as io
from torchvision import transforms
from PIL import Image
from sys import exit as e
import torchvision.models as models
from torchvision.utils import save_image
import copy
import numpy as np


from modules.vgg import VGG, StyleLoss, Normalization
import modules.util as util


def image_loader(image_name):
  imsize = 256
  loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
  image = Image.open(image_name)
  image = loader(image).unsqueeze(0)
  return image.type(torch.float)


def get_input_optimizer(input_img):
  # this line to show that input is a parameter that requires a gradient
  optimizer = optim.LBFGS([input_img.requires_grad_()], lr = 0.1)
  return optimizer


def get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std\
  , style_img, style_layers):
  cnn = copy.deepcopy(cnn)
  normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)
  style_losses = []

  model = nn.Sequential(normalization)
  i = 0
  for n_child, layer in enumerate(cnn.children()):
    if isinstance(layer, nn.Conv2d):
      i+=1
      name = f'conv_{i}'
    elif isinstance(layer, nn.ReLU):
      name = f'relu_{i}'
      layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.BatchNorm2d):
      name = f'bn_{i}'
    elif isinstance(layer, nn.MaxPool2d):
      name = f'pool_{i}'
    else:
      # continue
      raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))


    model.add_module(name, layer)
    if name in style_layers:
      target_feature = model(style_img).detach()
      style_loss = StyleLoss(target_feature)
      model.add_module(f"style_loss_{i}", style_loss)
      style_losses.append(style_loss)

  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], StyleLoss):
      break

  model = model[:(i + 1)]

  return model, style_losses



def gram_minimization(configs):

  tank = image_loader("./input/tank.png").squeeze()
  tank_blended = image_loader("./input/tank_blended.png").squeeze()
  source_img = image_loader("./input/plane.png")
  target_img = image_loader("./input/sky.png")
  mask_img = image_loader("./input/mask_plane.png").squeeze()
  # source_img = (tank_blended * mask_img).unsqueeze(0)
  # target_img = (tank * mask_img).unsqueeze(0)
  gt_gradient = util.compute_gt_gradient(source_img, target_img, mask_img.detach().numpy(), None)
  input_img = torch.randn(target_img.shape)
  mask_img = mask_img.squeeze(0).repeat(3,1).view(3,source_img.size(2),source_img.size(3)).unsqueeze(0)

  content_layers_default = ['conv_4']
  style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
  grad_weight = float(configs["params"]["grad_wt"])
  cnn = models.vgg19(pretrained=True).features.eval()
  cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
  cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
  normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

  target_img = normalization(target_img)
  # mean_shift = util.MeanShift('cpu')
  # target_img = mean_shift(target_img)

  # input_img = target_img.clone()
  # model, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, \
  #   cnn_normalization_std, source_img, style_layers_default)
  optimizer = get_input_optimizer(input_img)
  mse = torch.nn.MSELoss()
  run = [0]
  while run[0] <= configs["params"]["epochs"]:

    def closure():
      blend_img = torch.zeros(target_img.shape)
      blend_img = input_img*mask_img + target_img*(mask_img-1)*(-1)

      pred_gradient = util.laplacian_filter_tensor(blend_img)
      grad_loss = 0
      for c in range(len(pred_gradient)):
          grad_loss += mse(pred_gradient[c], gt_gradient[c])
      grad_loss /= len(pred_gradient)
      grad_loss *= grad_weight

      loss = grad_loss
      optimizer.zero_grad()
      loss.backward()

      run[0] += 1
      if run[0] % 10 == 0:
        print("run {}:".format(run))
        print('Loss : {:4f}'.format(
            loss.item()))
        print()
        save_image(input_img.squeeze(), f"output/sample/{run[0]}.png")

      return loss

    optimizer.step(closure)

  # a last correction...
  input_img.data.clamp_(0, 1)
  blend_img = torch.zeros(target_img.shape)
  # blend_img_np = blend_img.transpose(1,3).transpose(1,2).data.numpy()[0]

  invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
  blend_img = input_img*mask_img + target_img*(mask_img-1)*(-1)
  save_image(blend_img.squeeze(), "./output/blend_img_shift.png")
  blend_img = invTrans(blend_img.squeeze()).unsqueeze(0)
  save_image(input_img.squeeze(), "./output/input_img.png")
  save_image(blend_img.squeeze(), "./output/blend_img_unshift.png")
  # io.imsave("./output/blend_img_np.png", blend_img_np.astype(np.uint8))




















