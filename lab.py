from kan_convs import KANConv2DLayer
from utils.lib import *
from models import yolo_v8_x, yolo_v8_s, yolo_v8_l, yolo_v8_m, yolo_v8_n
from utils.measure import time_run
from custom import CosineConv2D

pretrain_path = 'weights/v8_m.pth'
model         = yolo_v8_m(CosineConv2D, 80, pretrain_path=pretrain_path)
images        = torch.randn(8, 3, 640, 640)
model.load_pretrain()


print(model.net.p1[0].conv.weight.shape)