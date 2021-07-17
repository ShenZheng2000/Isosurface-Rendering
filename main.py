import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from collections import OrderedDict
import argparse

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### This function reads the txt files

### This function reads the png images



def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_mgrid(sidelen, text, dim = 2, other_dim = 3): # dim default is 2.
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)]) # (x,y） related tensor

    tensors_1 = tuple([torch.linspace(-0.875, -0.875, steps=sidelen)]) # first input
    tensors_2 = tuple([torch.linspace(-0.875, -0.875, steps=sidelen)]) # second input
    tensors_3 = tuple([torch.linspace(-0.875, -0.875, steps=sidelen)]) # third input

    tensors = tensors + tensors_1 + tensors_2 + tensors_3 # total tensor

    #print(len(tensors))

    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim + other_dim) # this 3 means another 3 coordinates, need to adjust when additional time step is considered
    print("grided image has size of", mgrid.size())

    return mgrid

def get_cameraman_tensor(sidelength, image):
    # print("camera data is", type(skimage.data.camera()))
    # print("the shape of the img is", skimage.data.camera().shape)
    # img = Image.fromarray(skimage.data.camera()) # this one gets the camera data


    img = Image.open(image).convert('L')  # <class 'PIL.PngImagePlugin.PngImageFile'>
    img = np.asarray(img) # <class 'numpy.ndarray'>
    #print("the shape of our img is", img.shape) # SHOULD be gray scale image
    img = Image.fromarray(img) # <class 'PIL.Image.Image'>

    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)

    return img


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class ImageFitting(Dataset):
    def __init__(self, sidelength, image, text, other_dim):
        super().__init__()
        img = get_cameraman_tensor(sidelength, image)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelen = sidelength, dim = 2, text = text, other_dim=other_dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels



if __name__ == "__main__":

    # parse arguments (input txt file and input image file)
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--input_image', type=str, default="tiny_vorts0008_normalize_dataset/vorts0008_render_001.png")
    parser.add_argument('--input_txt', type=str,  default="text001.txt")
    parser.add_argument('--output_shape', type = int, default=16) # the paper uses 256 for this one
    parser.add_argument('--other_dim', type=int, default=3)
    config = parser.parse_args()


    ### Fitting an image
    cameraman = ImageFitting(sidelength=config.output_shape,
                             image = config.input_image,
                             text = config.input_txt,
                             other_dim=config.other_dim)


    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=5, out_features=1, hidden_features=config.output_shape, # original: in feature is 2. hidden feature is 256,
                      hidden_layers=3, outermost_linear=True)
    img_siren.to(device)

    ### train the model
    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)

    for step in range(total_steps):
        print("model_input shape is", model_input.size())
        model_output, coords = img_siren(model_input)

        print("model_output shape is", model_output.size())
        print("ground truth shape is", ground_truth.size())

        loss = ((model_output - ground_truth) ** 2).mean()


        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            #fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig, axes = plt.subplots(1, 1, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(config.output_shape, config.output_shape).detach().numpy())
            axes[1].imshow(img_grad.norm(dim=-1).cpu().view(config.output_shape, config.output_shape).detach().numpy())
            axes[2].imshow(img_laplacian.cpu().view(config.output_shape, config.output_shape).detach().numpy())
            plt.show()
            plt.close()

        optim.zero_grad()
        loss.backward()
        optim.step()
