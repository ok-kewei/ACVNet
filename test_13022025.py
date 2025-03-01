# from __future__ import print_function, division
import argparse
import os
import pdb

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2

cudnn.benchmark = True
print("Number of GPUs Available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # -1 하면 cpu됨
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
# parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
# 메모리사용 128도 18568? 까지는 간다
parser.add_argument('--maxdisp', type=int, default=64, help='maximum disparity')

### 여기 고핌
parser.add_argument('--dataset', default='ourdata', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/home/kewei/ACVNet/data/oxford/", help='data path')
parser.add_argument('--testlist', default='./filenames/ourdata_test.txt', help='testing list')
# parser.add_argument('--datapath', default="/home/kewei/ACVNet/data/singapore/", help='data path')
# parser.add_argument('--testlist', default='./filenames/singaporedata_test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--attention_weights_only', default=False, type=str, help='only train attention weights')
parser.add_argument('--freeze_attention_weights', default=False, type=str, help='freeze attention weights parameters')
parser.add_argument('--loadckpt', default='./pretrained_model/sceneflow.ckpt',
                    help='load the weights from a specific checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--savepath', default="./ourdata_testoutput/", help='data path')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# model, ckpt
model = __models__[args.model](args.maxdisp, args.attention_weights_only, args.freeze_attention_weights)
del args.maxdisp, args.attention_weights_only, args.freeze_attention_weights

model = nn.DataParallel(model)
model.cuda()
state_dict = torch.load(args.loadckpt)
del args.loadckpt

model_dict = model.state_dict()
pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)
model.eval()

gc.collect()
torch.cuda.empty_cache()
# del model

# dataset
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

def draw_disparity(disparity_map):
    disparity_map = disparity_map.astype(np.uint8)
    norm_disparity_map = (
                255 * ((disparity_map - np.min(disparity_map)) / (np.max(disparity_map) - np.min(disparity_map))))
    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_MAGMA)



def test():
    for batch_idx, sample in enumerate(TestImgLoader):
        gc.collect()
        torch.cuda.empty_cache()

        imgL, imgR = sample['left'], sample['right']

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        gc.collect()
        torch.cuda.empty_cache()
        op = model(imgL, imgR)  # list
        # [pred_attention, pred0, pred1, pred2]

        output = op[0]
        output = output.cpu().detach().numpy()  # <class 'numpy.ndarray'>
        output = np.array(output[0])
        color_disparity = draw_disparity(output)
        file_name = sample["left_filename"][0].replace("left/", "")
        # print(file_name)
        cv2.imwrite(args.savepath + file_name, color_disparity)
        output.tofile(args.savepath + file_name  + '.raw')
        del imgL, imgR, op, output, color_disparity
        gc.collect()
        torch.cuda.empty_cache()

        # output=op[0]
        # output=output.cpu().detach().numpy() #<class 'numpy.ndarray'>
        # output = np.array(output[0])
        # color_disparity=draw_disparity(output)
        # cv2.imwrite(args.savepath+str(time.time())+'.png',color_disparity)
        # output.tofile(args.savepath+str(time.time())+'.raw')
        # del imgL, imgR, op, output, color_disparity
        # gc.collect()
        # torch.cuda.empty_cache()

if __name__ == '__main__':
    test()



# def draw_disparity(disparity_map):
#     """
#     Normalize and colorize a disparity map for visualization.
#
#     Args:
#         disparity_map (numpy.ndarray): Raw disparity map (float32 or float64).
#
#     Returns:
#         numpy.ndarray: Colored disparity visualization (BGR image).
#     """
#     # Ensure disparity map is float32
#     disparity_map = disparity_map.astype(np.float32)
#
#     # Compute min and max, avoiding division by zero
#     min_val = np.min(disparity_map)
#     max_val = np.max(disparity_map)
#     epsilon = 1e-6  # Small value to prevent division by zero
#
#     if max_val - min_val < epsilon:
#         print("⚠ Warning: Disparity map has almost no variation! Visualization may be incorrect.")
#
#     # Normalize to range [0, 255]
#     norm_disparity_map = 255 * (disparity_map - min_val) / (max_val - min_val + epsilon)
#
#     # Convert to uint8
#     norm_disparity_map = np.uint8(norm_disparity_map)
#
#     # Apply colormap for visualization
#     color_disparity = cv2.applyColorMap(norm_disparity_map, cv2.COLORMAP_MAGMA)
#
#     return color_disparity

# def compute_3d_pose(disparity_map, focal_length, baseline, principal_point):
#     """
#     Convert disparity map to 3D coordinates.
#
#     Args:
#         disparity_map (numpy.ndarray): Disparity map (float values in pixels).
#         focal_length (float): Camera focal length in pixels.
#         baseline (float): Baseline distance between cameras in meters.
#         principal_point (tuple): (c_x, c_y) Optical center.
#
#     Returns:
#         numpy.ndarray: 3D coordinates (X, Y, Z) for each pixel.
#     """
#     h, w = disparity_map.shape  # Image dimensions
#     c_x, c_y = principal_point
#
#     # Avoid division by zero by setting very small disparities to NaN
#     disparity_map[disparity_map <= 0] = np.nan
#
#     # Compute depth Z = (f * B) / d
#     Z = (focal_length * baseline) / disparity_map
#
#     # Generate coordinate grids
#     x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
#
#     # Compute X and Y coordinates
#     X = (x_grid - c_x) * Z / focal_length
#     Y = (y_grid - c_y) * Z / focal_length
#
#     # Stack to create (H, W, 3) array
#     points_3d = np.dstack((X, Y, Z))
#
#     return points_3d


# def compute_3d_pose(disparity_map, focal_length, baseline, principal_point):
#     """
#     Convert disparity map to 3D coordinates.
#
#     Args:
#         disparity_map (numpy.ndarray): Disparity map (H, W, 3), multiple disparity channels.
#         focal_length (float): Camera focal length in pixels.
#         baseline (float): Baseline distance between cameras in meters.
#         principal_point (tuple): (c_x, c_y) Optical center.
#
#     Returns:
#         numpy.ndarray: 3D coordinates (H, W, 3), where each pixel has (X, Y, Z) coordinates.
#     """
#     h, w, c = disparity_map.shape  # Image dimensions
#     c_x, c_y = principal_point
#
#     # Select the first channel (assuming it contains the main disparity values)
#     disparity = disparity_map[..., 0].astype(np.float32)  # Convert to float
#
#     # Avoid division by zero by setting invalid disparities to a small nonzero value
#     disparity[disparity <= 0] = np.nan  # Mark invalid disparities as NaN
#
#     # Compute depth: Z = (f * B) / disparity
#     Z = (focal_length * baseline) / disparity
#
#     # Replace NaN values in Z with zero (or another fallback value)
#     Z = np.nan_to_num(Z, nan=0.0)
#
#     # Generate coordinate grids
#     x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
#
#     # Compute 3D coordinates
#     X = (x_grid - c_x) * Z / focal_length
#     Y = (y_grid - c_y) * Z / focal_length
#
#     # Stack to create (H, W, 3) array representing (X, Y, Z)
#     points_3d = np.dstack((X, Y, Z))
#
#     return points_3d
