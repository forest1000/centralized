# encoding: utf-8
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch 
from PIL import Image
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.surface_distance import SurfaceDistanceMetric

from src.utils.draw import draw_mask_and_save
from src.utils.metric_logger import EMAMetricLogger
from src.evaluation.evaluation_strategy import EvaluationStrategy

class FundusEvalStrategy(EvaluationStrategy):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def validate(self, model, data_loader, device, save_dir) -> dict:
        return self.custom_eval(model, data_loader, device, "val", save_dir)

    def test(self, model, data_loader, device, save_dir) -> dict:
        return self.custom_eval(model, data_loader, device, "test", save_dir)
    
    def custom_eval(self, model, data_loader, device, prefix, save_dir) -> dict:
        metrics = self._run(model, data_loader, device, save_dir)
        metric_dict = {}
        for key, value in metrics.items():
            metric_dict[f"{prefix}/{key}"] = value
        return metric_dict
    
    @torch.no_grad()
    def _run(self, model, data_loader, device, save_dir=None):
        model.eval()
        # part = ['cup', 'disc']
        model.to(device=device)
        num_val_batches = len(data_loader)

        dice_metric = DiceMetric(include_background=True, reduction="none")
        jc_metric = MeanIoU(include_background=True, reduction="mean")
        hd_metric = HausdorffDistanceMetric(include_background=True, reduction="none", percentile=95)
        asd_metric = SurfaceDistanceMetric(include_background=True, reduction="none")

        # iterate over the validation set
        for image_paths, images, masks in tqdm(data_loader, total=num_val_batches, desc='Evaluation', unit='batch', leave=False):
            # move images and labels to correct device and type
            images = images.to(device=device)
            n_imgs = images.size(0)
            masks = masks.to(device=device)
            cup_mask = masks.eq(0).float()
            disc_mask = masks.le(128).float()
            masks = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)),dim=1)

            output = model(images)
            pred_label = torch.sigmoid(output).ge(0.5)

            # # Loop over the number of classes (cup and disc in this case)
            # reference from the code of the paper: 
            # DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets
            for i in range(pred_label.shape[1]):  
                if pred_label[:, i].float().sum() < 1e-4:
                    # If no significant foreground is detected, you might want to set pred_label to all zeros
                    pred_label[:, i] = torch.zeros_like(pred_label[:, i])
            
            dice_values = dice_metric(y_pred=pred_label, y=masks)
            jc_metric(pred_label, masks)
            hd_metric(y_pred=pred_label, y=masks)
            asd_metric(y_pred=pred_label, y=masks)
            # save the predicted results
            if save_dir is not None:
                for i in range(n_imgs):
                    img_path = image_paths[i]
                    parent_folder = Path(self.cfg['eval']['target_dir'])
                    img_name = Path(img_path).name.split(".")[0]
                    dice_value = dice_values[i].cpu().numpy()
                    saved_path = parent_folder / save_dir / f"{img_name}_dice_{dice_value.mean():.4f}.png"
                    Path(saved_path).parent.mkdir(parents=True, exist_ok=True)
                    img = Image.open(img_path)
                    img = np.array(img)
                    draw_mask_and_save(img, pred_label[i], saved_path)
                    
        dc = dice_metric.aggregate().mean(axis=0)
        cup_dc = dc[0].item()
        disc_dc = dc[1].item()
        avg_dc = dc.mean().item()

        jc = jc_metric.aggregate().item()

        hd_aggregated = hd_metric.aggregate()
        asd_aggregated = asd_metric.aggregate()

        zero_value_replace = 100
        for i in range(hd_aggregated.shape[1]):
            # mask of valid values
            hd_aggregated[:,i] = torch.where(torch.isinf(hd_aggregated[:,i]) | torch.isnan(hd_aggregated[:,i]), zero_value_replace, hd_aggregated[:,i])
            asd_aggregated[:,i] = torch.where(torch.isinf(asd_aggregated[:,i]) | torch.isnan(asd_aggregated[:,i]), zero_value_replace, asd_aggregated[:,i])

        hd = hd_aggregated.mean().item()
        asd = asd_aggregated.mean().item()

        metrics = {
            "cup_dc": cup_dc,
            "disc_dc": disc_dc,
            "avg_dc": avg_dc,
            "jc": jc,
            "hd": hd,
            "asd": asd,
        }
        dice_metric.reset()
        jc_metric.reset()
        hd_metric.reset()
        asd_metric.reset()
        return metrics