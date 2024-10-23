from segment_anything import SamPredictor, sam_model_registry
from get_bbox import get_bounding_box
import SimpleITK as sitk
import numpy as np
import os

device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

img_path = '/path/to/imagesTr'
label_path = '/path/to/labelsTr'
img_names = sorted(os.listdir(img_path))
label_names = sorted(os.listdir(label_path))

for img_name, label_name in zip(img_names, label_names):
    img = sitk.ReadImage(os.path.join(img_path, img_name))
    label = sitk.ReadImage(os.path.join(label_path, label_name))
    img_arr = sitk.GetArrayFromImage(img)
    label_arr = sitk.GetArrayFromImage(label)
    img_arr = np.transpose(img_arr, (2, 1, 0))
    label_arr = np.transpose(label_arr, (2, 1, 0))
    x_len,y_len,z_len = img_arr.shape
    bbox = get_bounding_box(label_arr)
    masks = []
    for z in range(0, bbox["z_min"]):
        masks.append(np.zeros((1, x_len, y_len)))
    for z in range(bbox["z_min"], bbox["z_max"] + 1):
        input_slice = img_arr[..., z]
        input_slice = np.expand_dims(input_slice, axis=-1)
        input_slice = (input_slice - input_slice.min())/(input_slice.max() - input_slice.min()+1)*255
        input_slice = input_slice.astype(np.uint8)
        input_slice = np.repeat(input_slice, 3, axis=-1)
        predictor.set_image(input_slice)
        input_box = np.array([bbox["y_min"], bbox["x_min"], bbox["y_max"], bbox["x_max"]])
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        masks.append(mask)
    for z in range(bbox["z_max"] + 1, z_len):
        masks.append(np.zeros((1, x_len, y_len)))
    masks = np.stack(masks, axis=0)
    masks = masks.squeeze(axis=1)
    masks = np.transpose(masks, (0, 2, 1))
    masks = masks.astype(np.uint8)
    masks = sitk.GetImageFromArray(masks)
    masks.CopyInformation(img)
    output_path = '/path/to/sam_infer/'
    output_name = label_name.split('.')[0] + '_sam.nii.gz'
    sitk.WriteImage(masks, os.path.join(output_path, output_name))