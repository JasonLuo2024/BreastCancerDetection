__author__ = "JasonLuo"
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
from pydicom.pixel_data_handlers import apply_windowing
import os
from PIL import Image



def get_pixels_with_windowing(dcm_file):
    im = pydicom.dcmread(dcm_file)

    data = im.pixel_array

    # This line is the only difference in the two functions
    data = apply_windowing(data, im)

    if im.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    else:
        data = data - np.min(data)

    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


def windowing(image, save_folder, patientID):
    try:
        image_name = f"{patientID}{os.path.basename(image).split('.')[0]}.png"
        pixels = get_pixels_with_windowing(image)
        plt.imsave(os.path.join(save_folder, image_name), pixels, cmap='gray', format='png')
    except Exception as e:
        print(e)



def reverse(patient_dir,side_target):
    for file in os.listdir(patient_dir):
        if file.endswith(".png"):
            image = Image.open(os.path.join(patient_dir, file))

            # Flip horizontally (mirror)
            mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            mirrored_image.save(os.path.join(side_target, file))
