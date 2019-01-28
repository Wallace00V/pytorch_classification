import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from dataset.data_inference import MyDataset
from torch.utils.data import DataLoader
from models import *

### dataset ###
traindata = DataLoader(MyDataset("/Users/zhang/Downloads/catdog_project/label_train.txt",train=True),batch_size=128,shuffle=True,num_workers=4)
valdata = DataLoader(MyDataset("/Users/zhang/Downloads/catdog_project/label_val.txt",train=False),batch_size=128,shuffle=True,num_workers=4)

### get model and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=False,num_classes=2)
model.to(device)
optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum=0.9,weight_decay=0.005)
criterion = nn.CrossEntropyLoss().to(device)

def train():
    writer = SummaryWriter(log_dir="./logs/{}".format(model.__class__.__name__))
    for epoch in range(20):
        if epoch%5==0 and epoch!=0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print("learning rate is: ",param_group['lr'])

        for iter,data in enumerate(traindata):

            iteration = epoch * len(traindata) + iter

            model.train()
            img,label=data
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss.data.cpu().numpy(), iteration)
            print("epoch:" + str(epoch) + " " + "iterations:" + str(iter)+ " "+"loss: ",loss.data.cpu().numpy())

            model_name = "{}_iterations_{}.pth".format(model.__class__.__name__, iteration)
            model_dir = "./checkpoints/{}".format(model_name)
            if iteration%100==0:
                torch.save(model.state_dict(), model_dir)
                test(valdata,writer,iteration)

    print("Finished Training")
    writer.close()


def test(valdata,writer,iteration):

    model.eval()
    corrects=0
    total=0
    for img,label in valdata:
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        _, preds = torch.max(output, 1)
        corrects += torch.sum(preds == label)

        total+=img.size(0)

    writer.add_scalar('val/accuracy',corrects.data.cpu().numpy()/total,iteration)



train()