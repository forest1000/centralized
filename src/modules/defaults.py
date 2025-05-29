#
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import weakref
import copy
import os
import wandb

from shutil import copyfile
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.device_selector import get_free_device_name
from src.utils.metric_logger import MetricLogger, EMAMetricLogger
from src.tasks.task_registry import TaskRegistry

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}

class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    """
    def __init__(self, args, cfg) -> None:
        self._hooks: List[HookBase] = []

        self.factory = TaskRegistry.get_factory(cfg['task'])
        self.evaluation_strategy = self.factory.create_evaluation_strategy(cfg)
        self.args = args
        self.val_interval = self.args.val_interval
        self.cfg = copy.deepcopy(cfg)
        self.init_dataloader()
        self.setup_train()

        self.name = "defaults"
        self.metric_logger = MetricLogger()
        self.metric_test_logger = MetricLogger()
        self.metric_val_logger = MetricLogger()

        self.loss_logger = EMAMetricLogger()
        if self.cfg['hooks']['wandb']:
            self.wandb_init()

    def setup_train(self):
        self.iter = 0
        self.epoch = 0
        self.max_iter = self.cfg["train"]["max_iter"]
        self.max_epoch = self.cfg['train']['max_epoch']
        if self.cfg["train"]["device"] is not None:
            self.device = self.cfg["train"]["device"]
        else:
            gpu_exclude = self.cfg["train"]["gpu_exclude"]
            self.device = get_free_device_name(gpu_exclude_list=gpu_exclude)
            
        self.build_model()
        self.model = self.model.to(self.device)

        Path(self.cfg["train"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        # Copy config file to checkpoint folder
        self.cfg_file = os.path.join(self.cfg["train"]["checkpoint_dir"], os.path.basename(self.args.config))
        copyfile(self.args.config, self.cfg_file)

        self.build_optimizer()
        self.build_schedular(self.optimizer)

    def build_model(self):
        raise NotImplementedError
            
    def build_optimizer(self):
        raise NotImplementedError

    def build_schedular(self, optimizer):
        raise NotImplementedError

    def init_dataloader(self):
        raise NotImplementedError

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)
        
    def train(self, iter: int):
        """
        Args:
            iter (int): 
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training")
       
        with tqdm(total=iter) as pbar:
            try:
                self.before_train()
                self.before_step()
                self.run_step()
                self.after_step()
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': self.metric_logger.loss})
                    
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:

                self.after_train()    

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError
    
    def wandb_init(self):
        wandb.login(key=self.cfg["wandb"]["key"])
        self.wandb_id = wandb.util.generate_id()
        self.experiment = wandb.init(project=f'{self.cfg["wandb"]["project"]}', resume='allow', id=self.wandb_id, name=self.cfg["wandb"]["run_name"])
        self.experiment.config.update(
            dict(steps=self.cfg["train"]["max_iter"], batch_size=  self.cfg["train"]["batch_size"]*2,
                    learning_rate = self.cfg["train"]["optimizer"][self.opt_name]["lr"]), allow_val_change=True)
        
        logging.info("######## Running wandb logger")
    
