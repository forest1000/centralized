import logging
import torch 
import numpy as np
import copy
from tqdm import tqdm
import torch.nn.functional as F

from src.modules.defaults import TrainerBase
from src.modules import hooks
from src.model.unet import UNet
from src.tasks.task_registry import TaskRegistry
from src.datasets.sampler import TrainingSampler
from monai.losses import DiceCELoss, MaskedDiceLoss
from .semi_trainer import SemiTrainer

class preFixmatchTrainer(SemiTrainer):
    def __init__(self, args, cfg, is_fully_supervised=True) -> None:
        super().__init__(args, cfg)

    