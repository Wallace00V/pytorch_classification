import os, sys
import numpy as np
from PIL import Image
import cv2

from torch.utils.data.dataset import Dataset
from torchvision import transforms as T


def opencv_readimg(path):
    img = cv2.imread(path)
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return image

def pil_readimg(path):
    return Image.open(path)

class MyDataset(Dataset):
    def __init__(self, txt_path, train=True):
        f=open(txt_path)
        lines = f.readlines()
        self.len=len(lines)
        self.imgs_path=[]
        self.labels=[]
        for l in lines:
            self.imgs_path.append(l.split()[0])
            self.labels.append(int(l.split()[1]))
        if train is True:
            self.transform= T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(30),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img = pil_readimg(self.imgs_path[item])
        #img = opencv_readimg(self.imgs_path[item])
        img = self.transform(img)
        label = self.labels[item]
        return img,label

if __name__ == "__main__":
    # path = "/Users/zhang/Downloads/cat_train/cat.12455.jpg"
    # image = Image.open(path)
    # print(image.size)
    #
    # image.convert('BGR')
    txt = "/Users/zhang/Downloads/catdog_project/label_train.txt"
    dataset = MyDataset(txt,train=True)
    image = dataset[33][0]
    image.show()
    print("ok")