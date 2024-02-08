import torchvision
import torch
import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('--model_path', type=str, default = "/root/tool_chain_temporary/best.pth", help='The path of model save')
    parser.add_argument('--outdir', type=str, default = "/root/tool_chain_temporary", help='The path of model save')
    args = parser.parse_args()
    return args

def main(args):
  model = torchvision.models.resnet18(pretrained=False)
  model.fc = torch.nn.Linear(512,2)
  model.load_state_dict(torch.load(args.model_path))
  device = torch.device('cpu')
  model = model.to(device)
  model.eval()
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  torch_out = model(x)
  torch.onnx.export(model,
                    x,
                    args.outdir + "/resnet18_track_detection.onnx",
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'])

if __name__ == '__main__':
  args = get_args()
  main(args)