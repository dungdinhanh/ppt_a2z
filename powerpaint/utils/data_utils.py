
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image



def resize_and_center_crop(pil_image, target_width, target_height, mask=False):
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    if not mask:
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    else:
        resized_image = pil_image.resize((resized_width, resized_height), Image.NEAREST)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image