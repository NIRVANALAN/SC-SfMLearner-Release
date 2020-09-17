import torch
from torchsummary.torchsummary import summary
import torchvision
from models import FusionUNet
import traceback

torch.autograd.set_detect_anomaly(True)

N = 1
INPUT_SHAPE = (512, 512)
INPUT_NF = 6
Dm = torch.randn(N, 6, *INPUT_SHAPE)
Ds = torch.rand_like(Dm)
Dm.shape

up_conv1_inplanes = [192, 512, 768, 1280, 1024]
up_conv1_outplane = [128, 256, 512, 512, 512]

DFNet = FusionUNet(INPUT_NF, up_conv1_inplanes,
                   up_conv1_outplane, deepBlender=False)
# INPUT = [(6, *INPUT_SHAPE), (6, *INPUT_SHAPE)]
INPUT = [(6, *INPUT_SHAPE), (6, *INPUT_SHAPE)]
print(DFNet)

print(DFNet(Dm, Ds).shape)
traceback.print_exc()

#forward
# summary(DFNet, [(6, *INPUT_SHAPE), (6, *INPUT_SHAPE)])
