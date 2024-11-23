import os
import sys
import pandas as pd
import random
import logging

from glob import glob

import yaml
from pathlib import Path
from PIL import Image
import math
from datetime import datetime

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

filepath = Path(__file__).resolve().parent

details_dir = Path(current_dir).joinpath("details")
details_dir.mkdir(parents=True, exist_ok=True)
details_file = details_dir.joinpath("details.txt")
now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
"""
# Description: This script generates the labeled/unlabeled split for the clients in the federated semi-supervised learning setting.
# 1. Fundus Task
config = yaml.safe_load(open(filepath.joinpath("../configs/fundus/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/fundus_dofe/semi")
labeled_slice_rate = config.get("labeled_slice_rate", 0.05)

domain_names = {1:'DGS', 2:'RIM', 3:'REF', 4:'REF_val'}
# Prepare details content
details_content = f"Now Time: {now_time}\n"
details_content += "- dataset name: Fundus\n"
details_content += f"  - ratio: {labeled_slice_rate}\n"

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    fully_train_df = pd.read_csv(f'{client_folder}/data.csv')
    labeled_slice_num = math.ceil(len(fully_train_df) * labeled_slice_rate)
    labeled_df = fully_train_df.sample(n=labeled_slice_num, random_state=1)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)
    unlabeled_df = fully_train_df.drop(labeled_df.index)
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    
    details_content += f"\n    - {domain}: {labeled_slice_num}"

# 2. Prostate Task
config = yaml.safe_load(open(filepath.joinpath("../configs/prostate/scripts_conf.yaml")))
target_folder = config.get("target_folder", "~/prostate_mri/semi")
labeled_slice_rate = config.get("labeled_slice_rate", 0.05)
domain_names = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}
details_content += "\n\n- dataset name: Prostate\n"
details_content += f"  - ratio: {labeled_slice_rate}\n"

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    fully_train_df = pd.read_csv(f'{client_folder}/data.csv')
    labeled_slice_num = math.ceil(len(fully_train_df)*labeled_slice_rate)
    # notice: labeled data should be selected from consecutive slices within the same case
    all_cases = list(set(fully_train_df['image_id'].str.extract(r'(Case\d+)_')[0].to_list()))
    # Randomly shuffle cases
    random.shuffle(all_cases)

    labeled_slices = []
    total_slices = 0
    
    for case in all_cases:
        # Select all slices from the current case
        case_slices = fully_train_df[fully_train_df['image_id'].str.startswith(case)]
        
        # Check if adding these slices exceeds the required labeled_slice_num
        if total_slices + len(case_slices) > labeled_slice_num:
            remaining_slices = labeled_slice_num - total_slices
            labeled_slices.append(case_slices.sample(n=remaining_slices, random_state=1))
            breakabeled_
        else:
            labeled_slices.append(case_slices)
            total_slices += len(case_slices)

    labeled_df = pd.concat(labeled_slices)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)
    unlabeled_df = fully_train_df.drop(labeled_df.index)
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    
    details_content += f"\n    - {domain}: {labeled_slice_num}"
"""
# 3. Cardiac Task
config = yaml.safe_load(open(filepath.joinpath("../configs/cardiac/scripts_conf.yaml")))
target_folder = config.get("target_folder", "~/cardiac/semi")
labeled_slice_num = config.get("labeled_slice_number", 30)
domain_names = {1:'A', 2:'B', 3:'C', 4:'D'}
details_content = "\n- dataset name: Cardiac\n"

for domain in domain_names.keys():
    
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    fully_train_df = pd.read_csv(f'{client_folder}/data.csv')
    # notice: labeled data should be selected from consecutive slices within the same case
    all_cases = list(set(fully_train_df['image_id'].str.split(r'_').apply(lambda x: x[0]).to_list()))
    # Randomly shuffle cases
    random.shuffle(all_cases)

    labeled_slices = []
    total_slices = 0
    
    for case in all_cases:
        # Select all slices from the current case
        case_slices = fully_train_df[fully_train_df['image_id'].str.startswith(case)]
        
        # Check if adding these slices exceeds the required labeled_slice_num
        if total_slices + len(case_slices) > labeled_slice_num:
            remaining_slices = labeled_slice_num - total_slices
            labeled_slices.append(case_slices.sample(n=remaining_slices, random_state=1))
            break
        else:
            labeled_slices.append(case_slices)
            total_slices += len(case_slices)

    labeled_df = pd.concat(labeled_slices)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)
    unlabeled_df = fully_train_df.drop(labeled_df.index)
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    details_content += f"\n    - {domain}: {labeled_slice_num}"

# 1. Spinal Task
config = yaml.safe_load(open(filepath.joinpath("../configs/spinal/scripts_conf.yaml")))

target_folder = config.get("target_folder", "~/spinal/semi")
labeled_slice_num = config.get("labeled_slice_number", 30)

domain_names = {1:'site1', 2:'site2', 3:'site3', 4:'site4'}
# Prepare details content
details_content = '\n'
details_content += f"Now Time: {now_time}\n"
details_content += "- dataset name: Spinal\n"
details_content += f"  - ratio: {labeled_slice_num}\n"

for domain in domain_names.keys():
    client_folder = f'{target_folder}/client_{domain}'
    # generate the labeled/unlabeled split
    fully_train_df = pd.read_csv(f'{client_folder}/labeled_data.csv')

    labeled_df = fully_train_df.sample(n=labeled_slice_num, random_state=5)
    labeled_df.to_csv(f'{client_folder}/labeled.csv', index=False)
    unlabeled_df = fully_train_df.drop(labeled_df.index)
    unlabeled = pd.read_csv(f'{client_folder}/unlabeled_data.csv')
    unlabeled_df = pd.concat([unlabeled_df, unlabeled])
    unlabeled_df.to_csv(f'{client_folder}/unlabeled.csv', index=False)
    
    details_content += f"\n    - {domain}: {labeled_slice_num}"

details_content += "\n========================================================================\n\n"
with open(details_file, 'a') as f:
    f.write(details_content)