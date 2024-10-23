import numpy as np
import SimpleITK as sitk

def get_bounding_box(segmentation_mask):
    # Find the indices of the non-zero elements
    coords = np.argwhere(segmentation_mask)
    
    # If no object is found, return None
    if coords.size == 0:
        return None
    
    # Get the minimum and maximum indices along each dimension
    x_min, y_min, z_min = np.min(coords, axis=0)
    x_max, y_max, z_max = np.max(coords, axis=0)
    
    # Create the bounding box coordinates
    bounding_box = {
        'x_min': int(x_min),
        'x_max': int(x_max),
        'y_min': int(y_min),
        'y_max': int(y_max),
        'z_min': int(z_min),
        'z_max': int(z_max)
    }
    
    return bounding_box
