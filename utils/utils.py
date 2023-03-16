import glob
import math
import os
import random
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path
from sys import platform
from PIL import Image,ImageDraw,ImageFont
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)
def change_cv2_draw(image,strs,local,sizes,colour):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    # font = ImageFont.truetype("./data/simhei.ttf",sizes, encoding="utf-8")
    # draw.text((0,0),strs,colour,font=font)
    font = ImageFont.truetype("simsun.ttc", 32, encoding="unic")
    draw.text(local, strs, 'white', font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        image=change_cv2_draw(img,label,(int(c1[0]), int(c1[1]) - 30),5,[225,225,225])
    return image




