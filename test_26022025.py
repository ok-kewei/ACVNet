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
import matplotlib.pyplot as plt

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

# global point
point = (30, 320)

def draw_disparity1(disparity_map):
    disparity_map = disparity_map.astype(np.uint8)
    norm_disparity_map = (
                255 * ((disparity_map - np.min(disparity_map)) / (np.max(disparity_map) - np.min(disparity_map))))
    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_MAGMA)


def show_distance(event, x,y, args, params):
    global point
    # point = (x,y)
    # print(x,y)
    # global point
    if event == cv2.EVENT_MOUSEMOVE:  # Update position when the mouse moves
        point = (x, y)

    # print(depth)

def compute_depth_from_disparity(disparity_map, focal_length, baseline):
    """
    Convert disparity map to depth information.

    Args:
        disparity_map (numpy.ndarray): Disparity map (float values in pixels).
        focal_length (float): Camera focal length in pixels.
        baseline (float): Baseline distance between cameras in meters.

    Returns:
        numpy.ndarray: Depth map (Z values in meters).
    """
    # Avoid division by zero for invalid disparity values (e.g., zeros or negative values)
    disparity_map = np.nan_to_num(disparity_map, nan=0.0)  # Replace NaN values with 0

    # Compute depth: Z = (f * B) / d
    depth_map = (focal_length * baseline) / (disparity_map )

    # Optionally, you can handle very large depth values (due to small disparity) by setting a maximum depth
    depth_map[depth_map > 1000] = 1000  # Example: limit depth values to 1000 meters for visualization

    return depth_map

def compute_3d_from_disparity(disparity_map, focal_length, baseline):
    """
    Convert disparity map to depth information.

    Args:
        disparity_map (numpy.ndarray): Disparity map (float values in pixels).
        focal_length (float): Camera focal length in pixels.
        baseline (float): Baseline distance between cameras in meters.

    Returns:
        numpy.ndarray: Depth map (Z values in meters).
    """
    # Avoid division by zero for invalid disparity values (e.g., zeros or negative values)
    disparity_map = np.nan_to_num(disparity_map, nan=0.0)  # Replace NaN values with 0

    # Compute depth: Z = (f * B) / d
    depth_map = (focal_length * baseline) / (disparity_map )

    # Optionally, you can handle very large depth values (due to small disparity) by setting a maximum depth
    depth_map[depth_map > 1000] = 1000  # Example: limit depth values to 1000 meters for visualization

    return depth_map

def test():
    # OXFORD: Camera parameters (Modify these values based on calibration)
    focal_length = 983.044006  # Example: focal length in pixels
    baseline = 0.239983  # Example: baseline in meters
    principal_point = (643.646973, 493.378998)  # Example: image center for 1280x960 resolution

    # Extrinsic transformation (Modify based on your setup)
    R = np.array([[0.000000, -0.000000,  1.000000],  # Example identity rotation (no rotation)
                  [1.000000,  0.000000, -0.000000],
                  [0.000000,  1.000000,  0.000000]])

    T = np.array([0.0, 0.0, 0.0])  # Example translation in meters (camera position in world frame)

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

        # Normalize the disparity map for display
        disparity_map_norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        disparity_map_norm = np.uint8(disparity_map_norm)
         # Compute depth: Z = (f * B) / d
        depth_map = (focal_length * baseline) / (disparity_map_norm)
        # Compute real-world X, Y coordinates
        h, w = output.shape
        x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))

        X = (x_map - principal_point[0]) * depth_map / focal_length
        Y = (y_map - principal_point[1]) * depth_map / focal_length
        Z = depth_map  # Depth from stereo vision

        # depth_map = compute_depth_from_disparity(output, focal_length, baseline)
        print("depth_map.shape", depth_map.shape)
        print("depth_map", depth_map)
        print("depth_map [960, 1280]", depth_map[959, 1279])

        cv2.namedWindow("Depth Map")
        cv2.setMouseCallback("Depth Map", show_distance)

        while True:
            img_display = disparity_map_norm.copy()  # Copy original image to modify
            if point:
                x, y = point
                # text = f"({x},{y}) D:{depth_map[y,x]}"
                text = f"({x},{y}) D:{X[y,x], Y[y,x],Z[y,x]}"
                cv2.circle(img_display, (x, y), 5, (255, 0, 0), -1)  # Draw a red circle
                # cv2.putText(img_display, f"({x},{y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img_display, text,(x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Draw a red circle

            cv2.imshow("Depth Map", img_display)

            key = cv2.waitKey(1)
            if key == 27:  # Press ESC to exit
                break

        # for i in range(depth_map.shape[1]):
        #     # cv2.circle(disparity_map_norm,(point), 20, (255,0,0),1 )
        #     # print(point)
        #     cv2.putText(disparity_map_norm,"{}, ".format(point), (point[0], point[1]), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 0), 1)
        #
        #
        # # print(point)
        # cv2.namedWindow("Depth Map")
        # cv2.setMouseCallback("Depth Map", show_distance)
        # cv2.imshow("Depth Map", disparity_map_norm)


        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        del imgL, imgR, op, output, disparity_map_norm
        # del imgL, imgR, op, output, color_disparity
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    test()

