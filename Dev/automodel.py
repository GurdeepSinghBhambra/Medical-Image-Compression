from LoadImageDataset import ImageDatasetHandler as idh
import tensorflow as tf
import numpy as np
import warnings
import time
import cv2
import os


def createSubsetDataset(data_obj, lines=None):
    #mention lines=None if you want to use the whole dataset, else,
    #mention the no of 'lines' out of which you want create train, test, val set
    #'lines' trims the whole dataset
    if(lines!=None):
        data_obj.train_test_split(validation=True,
                                  df=data_obj.getChunkOfFile(start_index=0, end_index=lines, mode="master"),
                                  test_size=0.15,
                                  validation_size=0.25,
                                  shuffle=True)
    else:
        data_obj.train_test_split(validation=True,
                                  test_size=0.15,
                                  validation_size=0.25,
                                  shuffle=True)


def getDatasetByMB(obj, mode, size):
    row_count=0
    total_image_size=0
    total_available_images = obj.getFileRows(mode=mode)
    img_x, img_y = None, None
    file_gen = obj.datasetGenerator(mode=mode, batch_size=1, norm=(0, 1), dtype='float32')
    while(total_image_size<=size):
        if(row_count == total_available_images):
            warnings.warn("Total images available are less than the requested size")
            break
        temp_img_x, temp_img_y = next(file_gen)
        if((temp_img_x.nbytes/1024/1024)*2 + total_image_size > size):
            break

        if(total_image_size == 0):
            if(temp_img_x.nbytes/1024/1024 > size):
                raise ValueError("Single image size is greater than the requested size")
            print("single image size (in MB):", temp_img_x.nbytes/1024/1024)
            img_x, img_y = temp_img_x.copy(), temp_img_y.copy()
        else:    
            img_x = np.concatenate((img_x, temp_img_x))
            img_y = np.concatenate((img_y, temp_img_y))
        total_image_size = total_image_size + (temp_img_x.nbytes/1024/1024) + (temp_img_y.nbytes/1024/1024)
        row_count = row_count+1
    
    return img_x, img_y


def datasetInfo(dataset_obj):
    print("\nDATASET INFO:\n")
    sample_image = dataset_obj.getImages(df=dataset_obj.getChunkOfFile(0, 1, mode="master"))
    print("Image Info:")
    print("Image shape:", sample_image.shape)
    print("Image dtype:", sample_image.dtype)
    print("\nDataset Sizes:")
    print("Total Dataset size:", dataset_obj.getFileRows(mode="master"))
    print("Train Dataset size:", dataset_obj.getFileRows(mode="train"))
    print("Validation Dataset size:", dataset_obj.getFileRows(mode="validation"))
    print("Test Dataset size:", dataset_obj.getFileRows(mode="test"))
    print("\nSuitable Batch Sizes:", dataset_obj.getBatchSizes(), "\n")


def main():
    i = idh("d:\\Minor Project\\Dataset\\data")
    x, y = getDatasetByMB(i, 'master', 8)
    print("Total dataset in memory (in MB):", (x.nbytes/1024/1024)+(y.nbytes/1024/1024))
    print("Image shapes:", x.shape, y.shape)
    i.showImages(y)


if __name__ == "__main__":
    main()
