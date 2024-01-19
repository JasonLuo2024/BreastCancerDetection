__author__ = "JasonLuo"
import os
import pydicom
from tqdm import tqdm
from windowing import windowing
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def dicom_to_png(dicom_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_path)

    # Get the pixel data
    pixel_array = ds.pixel_array

    # Normalize the pixel values
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    pixel_range = pixel_max - pixel_min

    normalized_array = (pixel_array - pixel_min) / pixel_range * 255.0

    # Convert the pixel array to a PIL image
    image = Image.fromarray(normalized_array.astype('uint8'))

    # Crop the image based on the breast region
    return image

def getPNG(image, save_folder, patientID):
    try:
        image_name = f"{patientID}{os.path.basename(image).split('.')[0]}.png"
        pixels = dicom_to_png(image)
        plt.imsave(os.path.join(save_folder, image_name), pixels, cmap='gray', format='png')
    except Exception as e:
        print(e)

file_path = r'D:\BreastCancer\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM'
save_folder = r'D:\Breast_ROI\DDSM'

for item in tqdm(os.listdir(file_path)):
    patientId = item
    folder_path = os.path.join(file_path,item)
    for subpath, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                filePath = os.path.join(subpath, file)
                savePath = os.path.join(save_folder, item)
                os.makedirs(savePath, exist_ok=True)
                getPNG(filePath,savePath,patientId)