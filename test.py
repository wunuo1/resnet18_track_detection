import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np


# 创建一个torch.utils.data.Dataset的实现。因为模型输入为224*224，图像分辨率为960*224所以X方向坐标需要缩放
def get_x(value, width):
    """Gets the x value from the image filename"""
    return (float(int(value)) * 224.0 / 960.0 - width/2) / (width/2)

def get_y(value, height):
    """Gets the y value from the image filename"""
    return ((224 - float(int(value))) - height/2) / (height/2)

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory, random_hflips=False):
        print("1")
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory + "/image", '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        label_file = os.path.join(self.directory + "/label", os.path.basename(image_path)+".txt")
        with open('example.txt', 'r') as label_file:
            content = label_file.read()
            values = content.split()
            if len(values) == 2:
                value1 = int(values[0])
                value2 = int(values[1])
            else:
                print("文件格式不正确")
        print(value1,value2)
        x = float(get_x(value1, 224))
        y = float(get_y(value2, 224))
      
        if self.random_hflips:
          if float(np.random.rand(1)) > 0.5:
              image = transforms.functional.hflip(image)
              x = -x
        
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy().copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image,
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()

def main(args=None):
  # 需要根据自己的环境改为数据集存放位置
  train_dataset = XYDataset('./image_dataset/train', random_hflips=False)
  test_dataset = XYDataset('./image_dataset/test', random_hflips=False)

#   # 创建训练集和测试集
#   test_percent = 0.1
#   num_test = int(test_percent * len(dataset))
#   train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=24,
      shuffle=True,
      num_workers=0
  )

  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=24,
      shuffle=True,
      num_workers=0
  )
