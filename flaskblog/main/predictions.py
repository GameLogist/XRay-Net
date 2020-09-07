import os
import numpy as np
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt
from flaskblog.users.utils import save_picture, send_reset_email, save_prediction_picture

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func
import secrets
from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random

class DenseNet121(nn.Module):

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.densenet121(x)
        return x


class HeatmapGenerator():
 
    def __init__(self, pathModel, nnClassCount, transCrop):
       
        model = DenseNet121(nnClassCount).cuda()
        # model = DenseNet121(nnClassCount)
        use_gpu=True
        if use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model
        self.model.eval()
        
        self.weights = list(self.model.module.densenet121.features.parameters())[-2]

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize((transCrop, transCrop)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)  
        self.transformSequence = transforms.Compose(transformList)
     
    def generate(self, pathImageFile, pathOutputFile, transCrop, class_names, form_picture):

        with torch.no_grad():
            use_gpu=True
            flag=0
            print(pathImageFile)
            imageData = Image.open(pathImageFile).convert('RGB')
            image = cv2.equalizeHist(imgData)
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if use_gpu:
                imageData = imageData.cuda()
            l = self.model(imageData)
            output = self.model.module.densenet121.features(imageData)

            print("\n\n\n",l,"\n\n\n")
            # l2 = torch.ge(l,0.5)
            l = l.cpu()
            label = class_names[torch.max(l,1)[1]]
            # if torch.max(l,1)[0] > 0.5:
            #     flag_1 = 1
            # diseases = []
            # preds = []
            # l_num = list(np.squeeze(l.numpy()))

            # print("\n\n\n",l_num,"\n\n\n")
            
            # for i in l_num:
            #     if i > 0.2:
            #         # print("\n\n\n",np.where(l_num == i),"\n\n\n")
            #         diseases.append(class_names[l_num.index(i)])
            #         preds.append(i)
            
            # Z = [x for _,x in sorted(zip(preds,diseases), reverse=True)]
            # Z2 = preds.sort(reverse=True)
            # print("\n\n\n", diseases, "\n\n\n")
            # print("\n\n\n", preds, "\n\n\n")

            # print("\n\n\n", Z, "\n\n\n")
            # print("\n\n\n", Z2, "\n\n\n")

            # label = class_names[torch.max(l,1)[1]]
            heatmap = None
            for i in range (0, len(self.weights)):
                map = output[0,i,:,:]
                if i == 0: heatmap = self.weights[i] * map
                else: heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()
                
        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        
        img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        random_hex = secrets.token_hex(8)
        _, f_ext = os.path.splitext(pathImageFile)
        picture_fn = random_hex + f_ext
        picture_fn_full = 'E:/Minor/Code/app/FlaskApp/flaskblog/static/predictions/heatmaps/' + picture_fn
        cv2.imwrite(picture_fn_full, img)
        f2 = 'predictions/heatmaps/' + picture_fn
        
        return f2, label