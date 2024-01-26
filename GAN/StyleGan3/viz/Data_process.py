__author__ = "JasonLuo"
import pydicom
import os
import shutil
from tqdm.auto import tqdm
import numpy as np
from PIL import Image, ImageOps
from skimage import measure
import cv2

## this file is used to sort DCM file based on it viewPosition - CC and MLO
## and convert all the DCM images into PNG images
## then using LOG filter to upscale one image to six images
def multi_scale_analysis(image):
    image = np.array(image)
    scales = [ 5 ,7, 9,11,13 , 15]
    results = []
    for scale in scales:
        # Apply Laplacian operator with current scale
        filtered = cv2.Laplacian(image, cv2.CV_64F, ksize=scale)
        results.append(filtered)
    return results
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
    # cropped_image = crop_breast_region(image)

    return image



def crop_breast_region(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Apply thresholding to create a binary image
    binary_image = grayscale_image.point(lambda x: 0 if x < 128 else 255, '1')

    # Invert the binary image
    inverted_image = ImageOps.invert(binary_image)

    # Find contours in the inverted binary image
    contours = measure.find_contours(np.array(inverted_image), 0.5, fully_connected='low')

    # Find the largest contour (assumed to be the breast region)
    largest_contour = max(contours, key=len)

    # Get the bounding box coordinates of the largest contour
    min_row, min_col = np.min(largest_contour, axis=0)
    max_row, max_col = np.max(largest_contour, axis=0)

    # Crop the image based on the bounding box coordinates
    cropped_image = image.crop((min_col, min_row, max_col, max_row))


    return cropped_image

# Specify the directory containing the DICOM files


directory = 'D:/WholeDataSet'
target = 'D:/Estern_PNG'
for category in tqdm(['Abnormal']):
    path = os.path.join(directory, category)
    target_path = os.path.join(target, category)
    os.makedirs(path, exist_ok=True)
    os.makedirs(target_path , exist_ok=True)
    left_path = os.path.join(target_path, 'MLO')
    right_path = os.path.join(target_path, 'CC')
    for LR in [left_path, right_path]:
        os.makedirs(LR , exist_ok=True)
    invalid_count = 0
    for subpath in tqdm(os.listdir(path)):
        filePath = os.path.join(path,subpath)
        dicom_files = [os.path.join(filePath, filename) for filename in os.listdir(filePath) if filename.endswith('.dcm')]
        for file_path in dicom_files:
            dcm = pydicom.dcmread(file_path)
            try:
                viewposition = dcm[0x0018, 0x5101].value
                png_Image = dicom_to_png(file_path).resize((512, 512))

                base_name = os.path.splitext(subpath)[0] + '.png'
                save_path = right_path if viewposition in ["CC", "XCCL", "XCCM"] else left_path
                output_path = os.path.join(save_path, base_name)

                png_Image.save(output_path)


            except KeyError:
                invalid_count = invalid_count + 1
    print(invalid_count)



