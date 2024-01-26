We firstly applied Windowing, a medical imaging technique, to the mammography images. By optimizing the contrast between soft and dense tissues, this method significantly helps our CNN models to better differentiate and analyze different tissue regions. The utilization of this method was critical because it led to a fundamental improvement in the performance of our CNN models. Prior to windowing, the highest Fl score our models could achieve was less than 10%. 
Next, we utilized YOLOX, an open-source object detection algorithm, to precisely extract the regions of interest (RO Is) from the mammography images. The adoption of this advanced technique plays a crucial role in noticeably reducing the training loss. It also contributes to a considerable improvement in the accuracy of the models during testing phases, demonstrating the effectiveness of this approach in refining our model's performance. 
Lastly, to address the challenge of insufficent malignant cases and the imbalance of data, all right breast images were mirrored to create corresponding left breast images. This approach allows us to focus primarily on the view positions of the images, specifically Craniocaudal (CC) and Mediolateral Oblique (MLO), and minimize the potential negative impact of mammography laterality diversity. By doing so, we can effectively reduce the number of required CNN channels. This approach simplifies the model and ensures that each channel is trained on a representative dataset, improving the robustness and accuracy of our CNN models in breast cancer detection. 

2.2.1 Windowing 
In the context of DICOM images, it's common to encounter pixel values that exceed the 8-bit range (255). The process of normalizing the entire range down to 8 bits can lead to a loss of valuable information. To reduce information loss, we apply Window­ing, also known as contrast enhancement, to select a specific range of pixels from the original image before any normalization occurs, which enhances the contrast between soft and dense tissues, thereby improving the visual clarity and interpretability of the medical images. [8]

![Windowing](https://github.com/JasonLuo2024/BreastCancerDetection/assets/113136895/dc6026a3-6d3f-478e-a88e-072c828b3673)

2.2.2 Abstracting regions of interest(ROI) 
YOLOX is a supervised learning model utilized for object detection. Following the process of image windowing, we apply YOlOX to extract the regions of interest(ROI) from the breast images. 
Abstracting regions of interest (ROI) is a widely adopted technique in medical image analysis, allowing for focused examination and processing of specific areas within an image. This approach is particularly beneficial in mammography, where accurate diagnosis and assessment rely heavily on precise extraction of regions of interest (ROis). 

![Cropping](https://github.com/JasonLuo2024/BreastCancerDetection/assets/113136895/300f2244-1be7-4c79-afa5-bc7029efd2e3)

Mirroring, also referred to as Reflecting, is a common technique used in photography for creating a symmetrical version of an image by flipping it horizontally or vertically. This process involves creating a mirrored copy of an image, either horizontally or vertically. In our context, we utilize mirroring to address the challenge posed by mammography laterality and insufficent malignant cases. By converting all right breast images into left images, we simplifies the the categorization process, focusing only on the Left Craniocaudal (L-CC) and Left Mediolateral Oblique (L-MLO) view positions.
This simplifies the classification and reduces complexities associated with lateral differences in mammographic images.

![mirror](https://github.com/JasonLuo2024/BreastCancerDetection/assets/113136895/5eb587e4-c7ba-4b1b-a041-6876006f5f1a)
