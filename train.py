import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
import argparse

# 创建一个torch.utils.data.Dataset的实现。因为模型输入为224*224，图像分辨率为640*224所以X方向坐标需要缩放
def get_x(value, image_width):
    return (float(value) * 224.0 / float(image_width) - float(224/2)) / (224/2)

def get_y(value, image_height):
    return (float(224/2) - float(value) * 224.0 / float(image_height)) / (224/2)

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory, mean, stddev, random_hflips=False):
        self.directory = directory
        self.mean = mean
        self.stddev = stddev
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory + "/image", '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        image_width, image_height = image.size
        with open(os.path.join(self.directory + "/label", os.path.splitext(os.path.basename(image_path))[0]+".txt"), 'r') as label_file:
            content = label_file.read()
            values = content.split()
            if len(values) == 2:
                value1 = int(values[0])
                value2 = int(values[1])
            else:
                print("文件格式不正确")
        x = float(get_x(value1, image_width))
        y = float(get_y(value2, image_height))
      
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
                self.mean, self.stddev)
        return image, torch.tensor([x, y]).float()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train resnet18 (fc = 2)')
    parser.add_argument('--dataset', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--outdir', type=str, default = "/root/tool_chain_temporary", help='The path of model save')
    parser.add_argument('--batch-size' , type=int, default=24 ,help='Total batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--mean', nargs=3, type=float, default=(0.485, 0.456, 0.406), help='Mean')
    parser.add_argument('--stddev' , nargs=3, type=float, default=(0.229, 0.224, 0.225) ,help='Stddev')
    
    args = parser.parse_args()
    return args

def main(args):
  # 需要根据自己的环境改为数据集存放位置
  
  train_dataset = XYDataset(args.dataset + '/train', args.mean, args.stddev, random_hflips=False)
  test_dataset = XYDataset(args.dataset + '/test',  args.mean, args.stddev,random_hflips=False)
  print("train dataset path: " + args.dataset + '/train',flush=True)
  print("test dataset path: " + args.dataset + '/test',flush=True)

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

  # 创建ResNet18模型，这里选用已经预训练的模型，
  # 更改fc输出为2，即x、y坐标值
  model = models.resnet18(pretrained=True)
  model.fc = torch.nn.Linear(512, 2)
  device = torch.device('cpu')
  model = model.to(device)

  NUM_EPOCHS = args.epochs
  BEST_MODEL_PATH = args.outdir + '/best.pt'
  best_loss = 1e9

  optimizer = optim.Adam(model.parameters())

  for epoch in range(NUM_EPOCHS):
      
      model.train()
      train_loss = 0.0
      for images, labels in iter(train_loader):
          images = images.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          outputs = model(images)
          loss = F.mse_loss(outputs, labels)
          train_loss += float(loss)
          loss.backward()
          optimizer.step()
      train_loss /= len(train_loader)
      
      model.eval()
      test_loss = 0.0
      for images, labels in iter(test_loader):
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          loss = F.mse_loss(outputs, labels)
          test_loss += float(loss)
      test_loss /= len(test_loader)
      
      print('%f, %f' % (train_loss, test_loss),flush=True)
      if test_loss < best_loss:
          print("save",flush=True)
          torch.save(model.state_dict(), BEST_MODEL_PATH)
          best_loss = test_loss

if __name__ == '__main__':
    args = get_args()
    main(args)
