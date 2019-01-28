# pytorch_classification
使用pytorch(1.0)构建一个快速图像分类算法，并使用tensorboardx进行loss和accuracy可视化

dataset/：构建标签txt、准备dataloader

models/：存放网络模型定义

utils/：可用于网络可视化(CAM)、转caffe等，本仓库暂未使用

train.py：训练和验证模型

训练网络
```
python train.py
```

查看loss和accuracy
```
tensorboard --logdir=logs/
```


