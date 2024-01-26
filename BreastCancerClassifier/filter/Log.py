import numpy as np
from PIL import Image, ImageFilter
from PIL import Image
import cv2
import scipy.signal
def LaplacianOfGaussian(image):
    image = Image.open(image).convert('RGB')
    image = np.array(image)
    scales = [5, 7, 9, 11, 13, 15]
    results = {}

    for idx, scale in enumerate(scales):
        # Apply Laplacian operator with the current scale using PIL's ImageFilter
        laplacian_image = cv2.Laplacian(image, cv2.CV_64F, ksize=scale)
        laplacian_image = Image.fromarray(laplacian_image.astype(np.uint8)).convert('RGB')



        results[str(idx)] = laplacian_image
    return results



