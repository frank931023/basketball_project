import torch
from ultralytics.nn.tasks import PoseModel

torch.serialization.add_safe_globals([PoseModel])

ckpt = torch.load("models/court_keypoint_model_414.pt", map_location="cpu", weights_only=False)
print("成功讀取 checkpoint")
