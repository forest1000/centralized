import numpy as np
import nibabel as nib
from scipy.stats import mode
from PIL import Image

def load_and_vote_masks(mask_files):

    mask_list = []
    for mask_file in mask_files:
        current_mask = nib.load(mask_file).get_fdata().astype(int)
        mask_list.append(current_mask)
    
    mask_stack = np.stack(mask_list, axis=-1)
    final_mask, _ = mode(mask_stack, axis=-1)
    final_mask = final_mask.astype(np.int32)
    
    return final_mask