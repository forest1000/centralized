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

class CardiacEvalStrategy(EvaluationStrategy):
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
        part = ['lv', 'myo', 'rv'] 
        n_classes = len(part)
        model.to(device=device)
        num_val_batches = len(data_loader)

        dice_metric = DiceMetric(include_background=False, reduction="none", ignore_empty=False)
        jc_metric = MeanIoU(include_background=False, reduction="mean", ignore_empty=False)
        hd_metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
        asd_metric = SurfaceDistanceMetric(include_background=False, reduction="none")

        # iterate over the validation set
        for image_paths, image, mask in tqdm(data_loader, total=num_val_batches, desc='Evaluation', unit='batch', leave=False):
            # move images and labels to correct device and type
            image = image.to(device=device)
            n_imgs = image.size(0)
            mask = mask.to(device=device).long()
            output = model(image)
            
            mask = F.one_hot(mask, num_classes=output.shape[1]).permute(0, 3, 1, 2).float().to(device=device)
            
            pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            # Convert predicted label to one-hot encoding for segmentation
            # Shape of pred_one_hot: (batch_size, num_classes, height, width)
            pred_label = F.one_hot(pred_label, num_classes=output.shape[1]).permute(0, 3, 1, 2).float()

            # # Loop over the number of classes
            # reference from the code of the paper: 
            # DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets
            for i in range(pred_label.shape[1]):  
                if pred_label[:, i].float().sum() < 1e-4:
                    # If no significant foreground is detected, set pred_label to all zeros
                    pred_label[:, i] = torch.zeros_like(pred_label[:, i])

            dice_values = dice_metric(y_pred=pred_label, y=mask)
            jc_metric(y_pred=pred_label, y=mask)
            hd_metric(y_pred=pred_label, y=mask)
            asd_metric(y_pred=pred_label, y=mask)
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
                    draw_mask_and_save(img, pred_label[i,1:,:,:], saved_path)
                    
        dc = dice_metric.aggregate().mean(axis=0)
        lv_dc = dc[0].item()
        myo_dc = dc[1].item()
        rv_dc = dc[2].item()
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
            "lv_dc": lv_dc,
            "myo_dc": myo_dc,
            "rv_dc": rv_dc,
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