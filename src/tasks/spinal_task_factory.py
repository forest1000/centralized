import torch.nn as nn

from src.evaluation.spinal_eval import SpinalEvalStrategy
from src.tasks.task_factory import TaskFactory
from src.datasets.dataset_spinal import SpinalDataset

class SpinalTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg, **kwargs):
        return SpinalEvalStrategy(cfg)
    
    def create_dataset(self, mode, cfg, **kwargs):
        is_labeled = kwargs.get('is_labeled', False)
        return SpinalDataset(mode=mode, cfg=cfg, is_labeled=is_labeled)