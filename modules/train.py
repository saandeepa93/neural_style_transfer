import torch
from torch import nn, optim
import skimage.io as io
from torchvision import transforms
from PIL import Image
from sys import exit as e
import torchvision.models as models
from torchvision.utils import save_image
import copy


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
  optimizer = optim.LBFGS([input_img.requires_grad_()])
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

  # tank = image_loader("./input/tank.png").squeeze()
  # tank_blended = image_loader("./input/tank_blended.png").squeeze()
  # style_img = (tank * mask).unsqueeze(0)
  # grad_img = (tank_blended * mask).unsqueeze(0)
  # util.show(style_img.squeeze().permute(1, 2, 0).detach())
  # util.show(grad_img.squeeze().permute(1, 2, 0).detach())
  source_img = image_loader("./input/tank.png")
  target_img = image_loader("./input/field.png")
  mask_img = image_loader("./input/tank_mask.png").squeeze()
  gt_gradient = util.compute_gt_gradient(source_img, target_img, mask_img.detach().numpy(), None)
  input_img = torch.randn(target_img.shape)
  mask_img = mask_img.squeeze(0).repeat(3,1).view(3,source_img.size(2),source_img.size(3)).unsqueeze(0)

  content_layers_default = ['conv_4']
  style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
  grad_weight = 1e4
  cnn = models.vgg19(pretrained=True).features.eval()
  cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
  cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
  normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)
  target_img = normalization(target_img)
  print(target_img.size())

  # input_img = target_img.clone()
  # model, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, \
  #   cnn_normalization_std, source_img, style_layers_default)
  optimizer = get_input_optimizer(input_img)
  mse = torch.nn.MSELoss()
  run = [0]
  # while True:
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
      # input_img.data.clamp_(0, 1)
      # optimizer.zero_grad()
      # model(input_img)
      # style_score = 0

      # for sl in style_losses:
      #   style_score += sl.loss

      # style_score *= configs["params"]["style_wt"]

      loss = style_score
      loss.backward()

      run[0] += 1
      if run[0] % 10 == 0:
        print("run {}:".format(run))
        print('Style Loss : {:4f}'.format(
            style_score.item()))
        print()

      # if style_score.item() < 30:
      #   return style_score

      return style_score

    optimizer.step(closure)

  # a last correction...
  input_img.data.clamp_(0, 1)
  print(input_img.size())
  save_image(input_img.squeeze(), "./output/transferred.png")




















