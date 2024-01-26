import os
from tqdm import tqdm
import pydicom
import numpy as np
import cv2
import dicomsdl as dicoml


image_dir_pydicom = r'D:\Estern_Health\WholeDataSet\Normal'
image_dir_dicomsdl = r'D:\Breast_ROI\NL_Health'

os.makedirs(image_dir_pydicom, exist_ok=True)
os.makedirs(image_dir_dicomsdl, exist_ok=True)

train_images = []
for subpath, dirs, files in tqdm(os.walk(image_dir_pydicom)):
    for file in files:
        image_path = os.path.join(subpath,file)
        train_images.append(image_path)
print(len(train_images))


def process(f, save_folder=None, extension="png"):
    im = pydicom.dcmread(f)
    patient = f.split('\\')[-2]
    image_name = os.path.basename(f).split('.')[0]
    dataset = dicoml.open(f)
    img = dataset.pixelData()

    try:
        # Load only the variables we need
        center = dataset["WindowCenter"]
        width = dataset["WindowWidth"]
        bits_stored = dataset["BitsStored"]
        voi_lut_function = dataset["VOILUTFunction"]

        # For sigmoid it's a list, otherwise a single value
        if isinstance(center, list):
            center = center[0]
        if isinstance(width, list):
            width = width[0]

        # Set y_min, max & range
        y_min = 0
        y_max = float(2 ** bits_stored - 1)
        y_range = y_max

        # Function with default LINEAR (so for Nan, it will use linear)
        if voi_lut_function == "SIGMOID":
            img = y_range / (1 + np.exp(-4 * (img - center) / width)) + y_min
        else:
            # Checks width for < 1 (in our case not necessary, always >= 750)
            center -= 0.5
            width -= 1

            below = img <= (center - width / 2)
            above = img > (center + width / 2)
            between = np.logical_and(~below, ~above)

            img[below] = y_min
            img[above] = y_max
            if between.any():
                img[between] = (
                        ((img[between] - center) / width + 0.5) * y_range + y_min
                )
    except Exception as e:
        #         dataset = dicoml.open(img_path)
        img = dataset.pixelData()

    img = (img - img.min()) / (img.max() - img.min())

    if dataset["PhotometricInterpretation"] == "MONOCHROME1":
        img = 1 - img

    img = (img * 255).astype(np.uint8)

    file_name = os.path.join(save_folder,f"{patient}_{image_name}.{extension}")
    cv2.imwrite(file_name, img)



for f in train_images:
    process(f,save_folder=image_dir_dicomsdl, extension="png")
