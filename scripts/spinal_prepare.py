import os
import sys
import pandas as pd
import random
import numpy as np
from glob import glob
import nibabel as nib
from scipy.stats import mode

import yaml
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def load_and_vote_masks(mask_files):

    mask_list = []
    for mask_file in mask_files:
        current_mask = nib.load(mask_file).get_fdata().astype(int)
        mask_list.append(current_mask)
    
    mask_stack = np.stack(mask_list, axis=-1)
    final_mask, _ = mode(mask_stack, axis=-1)
    final_mask = final_mask.astype(np.int32)
    
    return final_mask

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/spinal/scripts_conf.yaml")))
target_folder = config.get("target_folder", "~/spinal/semi")
raw_data_foleder = config.get("raw_data_folder","~/spinal/spinal_data")
random.seed(config.get("seed",0))
resize = (config.get("resize")['width'], config.get("resize")['height'])
print(resize)
Path(target_folder).mkdir(parents=True, exist_ok=True)

#site1 process 
#"site1-sc{i}-image.nii.gz"  i from 01 to 10
#"site1-sc{i}-mask-r{j}.nii.gz" j from 1 to 4
img_files_train_site1 = glob(f'{raw_data_foleder}/training-data/site1-*image.nii.gz')
img_files_test_site1 = glob(f'{raw_data_foleder}/test-data/site1-*image.nii.gz')
x_diff, x_up, x_low= 0, 0, 0
y_diff, y_up, y_low = 0, 0, 0

df = pd.DataFrame(columns=[
    'image_id', 
    'image_train_path', 
    'image_vis_path',
    'segmentation_mask_path', 
])
for img_path in img_files_train_site1:
    case_id = img_path.split('-')[-2]
    print(img_path)
    mask_paths =[img_path.replace('image',f'mask-r{i}') for i in range(1,5)]
    img = nib.load(img_path).get_fdata()
    mask = load_and_vote_masks(mask_paths)

    target_img_train_folder = f'{target_folder}/client_1/data/{case_id}/img_train'
    target_img_vis_folder = f'{target_folder}/client_1/data/{case_id}/img_vis'
    target_mask_folder = f'{target_folder}/client_1/data/{case_id}/mask'
    target_mask_vis_folder = f'{target_folder}/client_1/data/{case_id}/mask_vis'
    Path(target_img_train_folder).mkdir(parents=True, exist_ok=True)
    Path(target_img_vis_folder).mkdir(parents=True, exist_ok=True)
    Path(target_mask_folder).mkdir(parents=True, exist_ok=True)
    Path(target_mask_vis_folder).mkdir(parents=True, exist_ok=True)
    #slice process
    for slice in range(img.shape[-1]):
        mask_slice = mask[:,:,slice].astype(np.int8)
        if not np.any(mask_slice):
            continue
        img_slice = img[:,:,slice]
        min_value = img_slice.min()
        max_value = img_slice.max() 

        img_slice_vis = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8).T

        img_slice_train = img_slice_vis.astype(np.float32)
        
        Image.fromarray(img_slice_vis).resize(resize, Image.Resampling.LANCZOS).save(f'{target_img_vis_folder}/{case_id}_{slice}.png')
        resize_img = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
        np.save(f'{target_img_train_folder}/{case_id}_{slice}.npy', resize_img)

        # Save mask_slice as PNG
        resize_mask = Image.fromarray(mask_slice.T).resize(resize, Image.Resampling.NEAREST)
        resize_mask.save(f"{target_mask_folder}/{case_id}_{slice}.png")
        resize_mask = np.array(resize_mask).astype(np.uint8)

        #calucuate padding ratio
        #x_diff and y_diff mean mask height and width
        r_mask = (resize_mask > 0)
        coords = np.column_stack(np.where(r_mask))

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        x_diff += x_max - x_min
        y_diff += y_max - y_min
        x_up += resize[0] - x_max
        x_low += x_min
        y_up += resize[1] - y_max
        y_low += y_min
        
        # Convert image to grayscale 
        value_mapping = {1: 128, 2:255}  
        mask_slice_vis = resize_mask.copy()

        for original_value, new_value in value_mapping.items():
            mask_slice_vis[resize_mask == original_value] = new_value

        # Resize and save mask_slice
        Image.fromarray(mask_slice_vis).save(f'{target_mask_vis_folder}/{case_id}_{slice}.png')

        row = pd.DataFrame([{
                'image_id': f'{case_id}_{slice}',
                'image_train_path': f'{target_img_train_folder}/{case_id}_{slice}.npy',
                'image_vis_path': f'{target_img_vis_folder}/{case_id}_{slice}.png',
                'segmentation_mask_path': f'{target_mask_folder}/{case_id}_{slice}.png',
            }])
        
        df = pd.concat([df, row], ignore_index=True)
df_number = len(df)
x_diff, x_up, x_low = x_diff/df_number , x_up/df_number , x_low/df_number 
y_diff, y_up, y_low = y_diff/df_number , y_up/df_number , y_low/df_number 
df.to_csv(f'{target_folder}/client_1/labeled_data.csv',index=False)

#for client1 unlabeled data
df_un = pd.DataFrame(columns=[
        'image_id', 
        'image_train_path', 
        'image_vis_path',
        'segmentation_mask_path'
        ])
for img_path in img_files_test_site1:
    case_id = img_path.split('-')[-2]
    print(img_path)
    img = nib.load(img_path).get_fdata()

    target_img_train_folder = f'{target_folder}/client_1/data/{case_id}/img_train'
    target_img_vis_folder = f'{target_folder}/client_1/data/{case_id}/img_vis'

    Path(target_img_train_folder).mkdir(parents=True, exist_ok=True)
    Path(target_img_vis_folder).mkdir(parents=True, exist_ok=True)

    #slice process
    for slice in range(img.shape[-1]):
        
        img_slice = img[:,:,slice]
        min_value = img_slice.min()
        max_value = img_slice.max() 

        img_slice_vis = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8).T

        img_slice_train = img_slice_vis.astype(np.float32)
        
        Image.fromarray(img_slice_vis).resize(resize, Image.Resampling.LANCZOS).save(f'{target_img_vis_folder}/{case_id}_{slice}.png')
        resize_img = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
        np.save(f'{target_img_train_folder}/{case_id}_{slice}.npy', resize_img)

        row = pd.DataFrame([{
                'image_id': f'{case_id}_{slice}',
                'image_train_path': f'{target_img_train_folder}/{case_id}_{slice}.npy',
                'image_vis_path': f'{target_img_vis_folder}/{case_id}_{slice}.png',
                'segmentation_mask_path': '',
            }])
        
        df_un = pd.concat([df_un, row], ignore_index=True)

df_un.to_csv(f'{target_folder}/client_1/unlabeled_data.csv',index=False)
print("Finish client1")
#Process for site 2, 3 and 4 using the information from site 1
crop_heights = {2:100,
               3:200,
               4:175,}
crop_widths = {2:100,
              3:200,
              4:175,
              }
for i in [3,4,2]:
    
    crop_height = crop_heights[i]
    crop_width = crop_widths[i]
    center_crop = transforms.CenterCrop((crop_height, crop_width))
    img_files = glob(f'{raw_data_foleder}/training-data/site{i}-*image.nii.gz')
    df = pd.DataFrame(columns=[
        'image_id', 
        'image_train_path', 
        'image_vis_path',
        'segmentation_mask_path'
        ])
    
    for img_path in img_files:
        case_id = img_path.split('-')[-2]
        print(img_path)
        mask_paths =[img_path.replace('image',f'mask-r{i}') for i in range(1,5)]
        img = nib.load(img_path).get_fdata()
        mask = load_and_vote_masks(mask_paths)

        target_img_train_folder = f'{target_folder}/client_{i}/data/{case_id}/img_train'
        target_img_vis_folder = f'{target_folder}/client_{i}/data/{case_id}/img_vis'
        target_mask_folder = f'{target_folder}/client_{i}/data/{case_id}/mask'
        target_mask_vis_folder = f'{target_folder}/client_{i}/data/{case_id}/mask_vis'
        Path(target_img_train_folder).mkdir(parents=True, exist_ok=True)
        Path(target_img_vis_folder).mkdir(parents=True, exist_ok=True)
        Path(target_mask_folder).mkdir(parents=True, exist_ok=True)
        Path(target_mask_vis_folder).mkdir(parents=True, exist_ok=True)
        #slice process
        for slice in range(img.shape[-1]):
            mask_slice = mask[:,:,slice].astype(np.int8)
            if not np.any(mask_slice):
                continue
            img_slice = img[:,:,slice]
            min_value = img_slice.min()
            max_value = img_slice.max() 

            img_slice_vis = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)

            #calucuate mask length
            remask = (mask_slice > 0).astype(np.uint8)
            coords = np.column_stack(np.where(remask > 0))
            if coords.size==0:
                continue
            else:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                x_d, y_d = x_max - x_min, y_max-y_min
                x_rate, y_rate = x_d/x_diff, y_d/y_diff
                x2_up = int(np.round(x_up * x_rate))
                x2_low = int(np.round(x_low * x_rate))
                y2_up = int(np.round(y_up * y_rate))
                y2_low = int(np.round(y_low * y_rate))

                # cropping
                mask_slice = mask_slice[y_min -y2_low: y_max + y2_up, x_min - x2_low: x_max + x2_up]
                img_slice_vis = img_slice_vis[y_min -y2_low: y_max + y2_up, x_min - x2_low: x_max + x2_up]


            img_slice_train = img_slice_vis.astype(np.float32).T
            Image.fromarray(img_slice_vis.T).resize(resize, Image.Resampling.LANCZOS).save(f'{target_img_vis_folder}/{case_id}_{slice}.png')
            resize_img = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
            np.save(f'{target_img_train_folder}/{case_id}_{slice}.npy', resize_img)

            # Save mask_slice as PNG
            resize_mask = Image.fromarray(mask_slice.T).resize(resize, Image.Resampling.NEAREST)
            resize_mask.save(f"{target_mask_folder}/{case_id}_{slice}.png")
            resize_mask = np.array(resize_mask).astype(np.uint8)

            r_mask = (resize_mask > 0)
            coords = np.column_stack(np.where(r_mask))

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            x_diff += x_max - x_min
            y_diff += y_max - y_min
            x_up += resize[0] - x_max
            x_low += x_min
            y_up += resize[1] - y_max
            y_low += y_min
            
            # Convert image to grayscale 
            value_mapping = {1: 128, 2:255}  
            mask_slice_vis = resize_mask.copy()

            for original_value, new_value in value_mapping.items():
                mask_slice_vis[resize_mask == original_value] = new_value

            # Resize and save mask_slice
            Image.fromarray(mask_slice_vis.T).save(f'{target_mask_vis_folder}/{case_id}_{slice}.png')

            row = pd.DataFrame([{
                    'image_id': f'{case_id}_{slice}',
                    'image_train_path': f'{target_img_train_folder}/{case_id}_{slice}.npy',
                    'image_vis_path': f'{target_img_vis_folder}/{case_id}_{slice}.png',
                    'segmentation_mask_path': f'{target_mask_folder}/{case_id}_{slice}.png',
                }])
            
            df = pd.concat([df, row], ignore_index=True)
    df.to_csv(f'{target_folder}/client_{i}/labeled_data.csv',index=False)
    
    #Here, it starts to crop for unlabeled data
    img_files = glob(f'{raw_data_foleder}/test-data/site{i}-*image.nii.gz')
    df_un = pd.DataFrame(columns=[
        'image_id', 
        'image_train_path', 
        'image_vis_path',
        'segmentation_mask_path'
        ])
    
    for img_path in img_files:
        case_id = img_path.split('-')[-2]
        print(img_path)
        img = nib.load(img_path).get_fdata()

        target_img_train_folder = f'{target_folder}/client_{i}/data/{case_id}/img_train'
        target_img_vis_folder = f'{target_folder}/client_{i}/data/{case_id}/img_vis'
        Path(target_img_train_folder).mkdir(parents=True, exist_ok=True)
        Path(target_img_vis_folder).mkdir(parents=True, exist_ok=True)
        #slice process
        for slice in range(img.shape[-1]):
            img_slice = img[:,:,slice]
            min_value = img_slice.min()
            max_value = img_slice.max() 
            if not np.any(img_slice):
   
                print(f"skip {case_id}_{slice}")
                continue

            img_slice_vis = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)

            #cropping
            img_pil = Image.fromarray(img_slice_vis)
            img_cropped = center_crop(img_pil)
            img_cropped = np.array(img_cropped).T

            img_slice_train = img_cropped.astype(np.float32)
            Image.fromarray(img_cropped).resize(resize, Image.Resampling.LANCZOS).save(f'{target_img_vis_folder}/{case_id}_{slice}.png')
            resize_img = np.array(Image.fromarray(img_slice_train).resize(resize, Image.Resampling.LANCZOS))
            np.save(f'{target_img_train_folder}/{case_id}_{slice}.npy', resize_img)

            row = pd.DataFrame([{
                    'image_id': f'{case_id}_{slice}',
                    'image_train_path': f'{target_img_train_folder}/{case_id}_{slice}.npy',
                    'image_vis_path': f'{target_img_vis_folder}/{case_id}_{slice}.png',
                    'segmentation_mask_path': '',
                }])
            
            df_un = pd.concat([df_un, row], ignore_index=True)
    df_un.to_csv(f'{target_folder}/client_{i}/unlabeled_data.csv',index=False)
        

print("Processe finish")

