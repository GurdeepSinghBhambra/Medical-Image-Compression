# Medical-Image-Compression
Transferring medical images from one center to another is common use in telemedicine. These high-quality images stored in DICOM format require higher bandwidth for transmission and large storage space in PACS (Picture Archiving and Communication System) memory. Therefore, reducing the image size by preserving diagnostic information has become a need. In this sense, medical image compression is a technique that overcomes both transmission and storage cost by suggesting lossy and lossless compression algorithms. High resolution medical images obtained by different imaging modalities stored in PACS needs higher storage space and bandwidth because of requiring much space in memory. Compression of medical images is important for efficient use of database. The main purpose of image compression is to reduce the number of bits representing the image while preserving the image quality and the intensity level of the pixels as much as possible depending on grayscale or RGB image. Since medical images also contain diagnostic information about a disease or an artifact, less or no loss of detail in terms of quality is desired while compressing the significant areas.

DEV OS: Windows 10 
Python Version: 3.6
Used anaconda to develop the project.
Hosted the website on aws ec2 ubuntu instance (For demonstartion purposes, use the 'app' directory for that).

Directories:
  Dev: This folder contains the code and python modules to extract, preprocess data. OIt also has the model training code.
  ENV: This folder contains the instruction and the env file to re-create the conda environment.
  app: This is the final project directory that has the trained model, web back-end and front-end files

PPT Link: https://prezi.com/view/0JHKM955MT83wGpHsSZv/

Other files are self-explanatory.

