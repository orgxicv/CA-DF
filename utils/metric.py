import numpy as np
from scipy.stats import entropy
from sklearn import metrics
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
import math
import torch.nn.functional as F
import os

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=256):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def forward(self, I, J, win=[15]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims

        sum_filt = torch.ones([1, 1, *win]).cuda()

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)

        # cc = cross * cross / (I_var * J_var + 1e-5)
        cc = torch.sum(cross) / torch.sqrt(torch.sum(I_var) * torch.sum(J_var))

        return -1 * torch.mean(cc)

class NMI:
    """
    Normalized Mutual Information (NMI) loss calculation.
    """
    from scipy.stats import entropy

    def information(self, tensorA, bins=256):
        # Convert tensor to numpy array
        imageA = tensorA.squeeze().detach().cpu().numpy()  # Remove batch dim and convert to numpy
        if imageA.max() <= 1:
            imageA = (imageA * bins).astype('uint8')  # Scale to 0-255 if in 0-1 range

        # Flatten image to 1D array and compute histogram
        hist, x_edges = np.histogram(
            imageA.ravel(),  # Flatten image
            bins=bins,
            density=True
        )

        # Calculate probability distribution
        p_x = hist / np.sum(hist)  # Normalize

        # Compute information (entropy)
        mi = np.sum(p_x * np.log2(p_x + 1e-10))  # Add small epsilon to avoid log(0)
        
        return mi
    
    def cross_information(self, tensorA, tensorB, bins=256):
        """
        Calculate mutual information between two tensors.

        Args:
            tensorA (torch.Tensor): First tensor with shape (1, H, W)
            tensorB (torch.Tensor): Second tensor with shape (1, H, W)
            bins (int): Number of histogram bins (default: 256)

        Returns:
            mi (float): Mutual information value between the tensors
        """
        # Convert tensors to numpy arrays
        imageA = tensorA.squeeze().detach().cpu().numpy()  # Remove batch dim and convert
        imageB = tensorB.squeeze().detach().cpu().numpy()
        if imageA.max() <= 1:
            imageA = (imageA * bins).astype('uint8')  # Scale if needed
            imageB = (imageB * bins).astype('uint8')

        # Compute 2D histogram of joint distribution
        hist_2d, x_edges, y_edges = np.histogram2d(
            imageA.ravel(),  # Flatten first image
            imageB.ravel(),  # Flatten second image
            bins=bins,
            density=True
        )

        # Calculate joint probability distribution
        p_xy = hist_2d / np.sum(hist_2d)  # Normalize

        # Compute mutual information
        mi = np.sum(p_xy * np.log2(p_xy + 1e-10))  # Add epsilon for numerical stability
        
        return mi

    def loss(self, y_true, y_pred):
        """
        Calculate NMI loss between true and predicted tensors.
        
        Args:
            y_true (torch.Tensor): Ground truth tensor
            y_pred (torch.Tensor): Predicted tensor
            
        Returns:
            torch.Tensor: NMI loss value
        """
        return torch.Tensor([self.information(y_true) + self.information(y_pred) - self.cross_information(y_true, y_pred)])