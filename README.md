# LicensePlateDetection
The aim of this project is to perform license plate detection and text extraction without using any deep learning methods.

This model has been made purely with OpenCv. It first preprocess the image, applies perspective correction and then uses easy ocr to extract text in the given image.

#Preprocessing Steps
First the image is converted from BGR to GrayScale. A Gaussian Blur is applied along with adaptive thresholding

#FindLicensePLate
The license plate is then found using contour detection and assuming a specific aspect ratio for the number plate

#Perspective Correction
This function crops out the background and only retains the license plate

Attaching some output photos

![Screenshot 2025-02-08 182042](https://github.com/user-attachments/assets/d2456a1c-1d07-47b4-8e61-51345d7a4887)
![Screenshot 2025-02-08 182049](https://github.com/user-attachments/assets/36492f98-2a0e-4923-9758-4383272a3dbe)
