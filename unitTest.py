import torch
from torchsummary.torchsummary import summary
from models import FusionUNet
import traceback

torch.autograd.set_detect_anomaly(True)

N = 1
INPUT_SHAPE = (512, 512)
INPUT_NF = 3

Dm = torch.randn(INPUT_NF, INPUT_NF, *INPUT_SHAPE)
Ds = torch.rand_like(Dm)
Dm.shape

up_conv1_inplanes = [192, 512, 768, 1280, 1024]
up_conv1_outplane = [128, 256, 512, 512, 512]

DFNet = FusionUNet(INPUT_NF, up_conv1_inplanes,
                   up_conv1_outplane, deepBlender=False)

DBNet_up_conv1_inplanes = [384, 768, 1280, 1280, 1024]
DBNet_up_conv1_outplane = [64, 256, 512, 512, 512]

DBNet = FusionUNet(INPUT_NF, DBNet_up_conv1_inplanes,
                   DBNet_up_conv1_outplane, deepBlender=True)
# INPUT = [(6, *INPUT_SHAPE), (6, *INPUT_SHAPE)]
# INPUT = [(INPUT_NF, *INPUT_SHAPE), (INPUT_NF, *INPUT_SHAPE)]
print(DBNet)

print(DBNet(Dm, Ds).shape)
traceback.print_exc()

#forward
# summary(DFNet, [(6, *INPUT_SHAPE), (6, *INPUT_SHAPE)])
