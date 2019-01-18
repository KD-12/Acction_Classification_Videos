
import numpy as np
import cv2
from collections import defaultdict
import glob
import os
from PIL import Image
import scipy
from scipy import io
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def preprocess():
    path = ".\\UCF101_dimitris_course\\UCF101_release\\images_class1"
    data_dict = defaultdict(list)
    transform_1 = transforms.Compose([
                                    transforms.FiveCrop((224,224)),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                    ])
    transform_2 = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for dirname in os.listdir(path):
        img_path = glob.glob(path+"\\"+dirname+"\\"+"*.jpg")
        for i, filename in enumerate(img_path):
            print(filename)
            img = Image.open(filename)
            img_transform = transform_2(img)
            img_transform = transforms.functional.to_pil_image(img_transform)
            img_transform = transform_1(img_transform)
            data_dict[dirname].append(img_transform)
    return data_dict


def get_features_vgg(model, dataloader):
    feature_path_root = "./UCF101_dimitris_course/UCF101_release/vgg16_relu6/"
    for dirname, imgs in dataloader.items():
        print(dirname)
        features = []
        for e, img in enumerate(imgs):
            x_var = Variable(img)
            output_scores = model(x_var.view(-1, 3, 224, 224))
            temp_feature = output_scores.view(1,5, -1).mean(1).detach().numpy()
            features.append(temp_feature)
        dir_features = np.transpose(np.vstack(features))
        scipy.io.savemat(feature_path_root+dirname+".mat", {"Feature":dir_features})

if __name__ == "__main__":
    print("Starting VGG feature extraction...")
    data_dict = preprocess()

    vgg_model = torchvision.models.vgg16(pretrained='imagenet')
    for child in list(vgg_model.classifier.children()):
            for param in list(child.parameters()):
                param.requires_grad = False
    features = list(vgg_model.classifier.children())[:-1]
    vgg_model.classifier = nn.Sequential(*features)

    get_features_vgg(vgg_model, data_dict)
    print("Validating shape of matrice written...")
    x = scipy.io.loadmat("./UCF101_dimitris_course/UCF101_release/vgg16_relu6/v_000002.mat")
    print(x["Feature"].shape)
