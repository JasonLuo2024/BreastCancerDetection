import torch
import glob
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
model = torch.hub.load(r'C:\Users\Public\github\breastcancerdetection\yolov5', 'custom', path=r'C:\Users\Public\github\breastcancerdetection\yolov5\ROI_models\rsna-roi-003.pt', source='local')

directory = r'D:\NL_Health_Dcm_PNG_Windowing'
image_list = []
for subpath, dirs, files in tqdm(os.walk(directory)):
    for file in files:
        image_path = os.path.join(subpath,file)
        image_list.append(image_path)
print(len(image_list))

images = []
for img_file in random.sample(image_list,
                              25):  # it is fixed to 25 random predictions - if you want to change it remeber to change plot_roi as well

    # Read file from file
    frame = cv2.imread(img_file)

    # Make prediction
    detections = model(frame)

    # Convert results to Pandas style
    results = detections.pandas().xyxy[0].to_dict(orient="records")

    # Plot result (in 99.99% it predicts only one instance - certainly you can assure that only best prediction is used)
    for result in results:
        images.append(
            cv2.rectangle(frame, (int(result['xmin']), int(result['ymin'])), (int(result['xmax']), int(result['ymax'])),
                          (255, 0, 0), 4))

    # Plot result
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

    for idx, image in enumerate(images):
        i = idx % 5
        j = idx // 5
        axes[i, j].imshow(image)

    plt.subplots_adjust(wspace=0, hspace=.2)
    plt.show()
    plt.savefig('output.png')

