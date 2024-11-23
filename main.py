import yaml
import logging
import os

import random
import numpy as np
import torch
from torch.backends import cudnn
from pathlib import Path
from datetime import datetime


from src.utils.args_parser import args, args2cfg
task_name = os.path.basename(os.path.dirname(args.config))
now = datetime.now().strftime('%Y%m%d_%H%M%S')
trainer_name = args.trainer.lower()
unseen_client = args.unseen_clients[0]
Path(f'./log/{task_name}/{unseen_client}/{trainer_name}').mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'log/{task_name}/{unseen_client}/{trainer_name}/{args.run_name}_{now}.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

from src.modules.supervised_trainer import SupervisedTrainer
from src.modules.semi_trainer import SemiTrainer
from src.modules.only_labeled_trainer import OnlylabelTrainer

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
if __name__ == "__main__":
    try:
        unseen_client = args.unseen_clients[0]
        cfg = yaml.safe_load(open(args.config))
        checkpoint = os.path.join(cfg['train']['checkpoint_dir'],
                                   f'experiment_client_{unseen_client}','checkpoints')
        
        cfg = args2cfg(cfg, args)
        cfg['dataset']['train'] = os.path.join(cfg['dataset']['train'],f'experiment_client_{unseen_client}')
        cfg['dataset']['test'] = os.path.join(cfg['dataset']['test'],f'experiment_client_{unseen_client}')
        #cfg['dataset']['val'] = os.path.join(cfg['dataset']['val'],f'experiment_client_{unseen_client}')
        if args.ema == 'True':
            cfg['train']['ema'] = True
       
        project_name = cfg['wandb']['project']
        if cfg['train']['ema'] and trainer_name == 'fixmatch':
            run_info = f'{trainer_name}_EMA_{now}'
        else:    
            run_info = f'{trainer_name}_{now}'
        cfg['eval']['target_dir'] = os.path.join(cfg['eval']['target_dir'],
                                                 f'experiment_client_{unseen_client}','result'
                                                 ,project_name,run_info)
        
        cfg['train']['checkpoint_dir'] =os.path.join(cfg['train']['checkpoint_dir'],
                                                 f'experiment_client_{unseen_client}','result'
                                                 ,project_name,run_info)
        train_path = cfg['dataset']['train'] 
        test_path = cfg['dataset']['test'] 
        logging.info(f"Training data path: {train_path}")
        logging.info(f'Test data path: {test_path}')
        max_iter = cfg['train']['max_iter']
        logging.info(f"Trainer_name : {trainer_name}")
        if trainer_name == 'fixmatch':
            cfg['wandb']['run_name'] = args.run_name
            if cfg['train']['ema']:
                cfg['wandb']['run_name'] = cfg['wandb']['run_name'] + "_EMA"
            trainer = SemiTrainer(args, cfg)
        elif trainer_name == 'only_label':
            trainer = OnlylabelTrainer(args, cfg)
        elif trainer_name == 'supervised':
            trainer = SupervisedTrainer(args, cfg)
        trainer.train(max_iter)
        
    except Exception as e:
        logging.critical(e, exc_info=True)


