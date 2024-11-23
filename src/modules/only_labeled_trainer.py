import torch 

from src.tasks.task_registry import TaskRegistry
from src.datasets.sampler import TrainingSampler
from .supervised_trainer import SupervisedTrainer

class OnlylabelTrainer(SupervisedTrainer):
    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        self.register_hooks(self.build_hooks())
        print(f'device : {self.device}')
    
    def init_dataloader(self):
        factory = TaskRegistry.get_factory(self.cfg['task'])
        
        
        self.cfg['dataset']['train'] = f"{self.cfg['dataset']['train']}/labeled.csv"
        dataset = factory.create_dataset(mode='train', is_labeled = True, cfg=self.cfg)

        self._train_data_num = len(dataset)
        batch_size = self.cfg["train"]["batch_size"] * 2
        self.iter_per_epoch = self._train_data_num // batch_size + 1        
        num_workers = self.cfg["train"]["num_workers"]
        seed = self.cfg["dataset"]["seed"]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                                    sampler = TrainingSampler(len(dataset),seed=seed))
        
        self._data_iter = iter(data_loader)