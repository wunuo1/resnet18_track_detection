import argparse
import torchvision
import torch
import PIL.Image
from PIL import ImageDraw
import torchvision.transforms as transforms

def detect(weights,source,output):
  model = torchvision.models.resnet18(pretrained=False)
  model.fc = torch.nn.Linear(512,2)
  model.load_state_dict(torch.load(weights))
  model.eval()
  device = torch.device('cpu')
  model = model.to(device)
  image_raw = PIL.Image.open(source)
  imagedraw = ImageDraw.Draw(image_raw)
  color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
  image = color_jitter(image_raw)
  image = transforms.functional.resize(image, (224, 224))
  image = transforms.functional.to_tensor(image)
  image = image.numpy().copy()
  image = torch.from_numpy(image)
  image = transforms.functional.normalize(image,
          [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  image = image.unsqueeze(dim=0)
  pred = model(image)
  pred = torch.squeeze(pred)

  x = int((pred[0] * 112 + 112) * 640.0 / 224.0)
  y = int(224 - (pred[1] * 112 + 112))
  print(x,y)
  # imagedraw.point((100,100),)
  for i in range(x, x + 10):
    for j in range(y, y + 10):
        imagedraw.point((i, j), (255,0,0))
  image_raw.save(output)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best_line_follower_model_xy.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='image.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output_image.jpg', help='output folder')  # output folder
    args = parser.parse_args()

    with torch.no_grad():
        detect(args.weights,args.source,args.output)