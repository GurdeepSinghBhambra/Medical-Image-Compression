from LoadImageDataset import ImageDatasetHandler as idh
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
#import numpy as np
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

def encoder_model():
    enc = tf.keras.models.Sequential(name="Encoder")
    enc.add(tf.keras.layers.Input(shape=(1024, 1024, 1), 
                                name="Encoder_Input"))
    enc.add(tf.keras.layers.Conv2D(filters=16, 
                                   kernel_size=(2, 2), 
                                   activation="relu",
                                   padding="same",
                                   data_format="channels_last", 
                                   name="Encoder_Conv2D_1"))
    enc.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="Encoder_MaxPooling2D_1"))
    enc.add(tf.keras.layers.Conv2D(filters=8, 
                                   kernel_size=(4, 4), 
                                   activation="relu",
                                   padding="same",
                                   data_format="channels_last", 
                                   name="Encoder_Conv2D_2"))
    enc.add(tf.keras.layers.Conv2D(filters=4,
                                   kernel_size=(2, 2),
                                   activation="relu",
                                   padding="same", 
                                   data_format="channels_last", 
                                   name="Encoder_Conv2D_3"))
    enc.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="Encoder_MaxPooling2D_2"))
    enc.add(tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(2, 2),
                                   activation="sigmoid",
                                   padding="same", 
                                   data_format="channels_last", 
                                   name="Encoder_Conv2D_4"))
    enc.add(tf.keras.layers.Flatten(name="Encoder_Flatten_Output"))
    return enc

#Ratio: 16:1

def decoder_model():
    dec = tf.keras.models.Sequential(name="Decoder")
    dec.add(tf.keras.layers.Input(shape=(65536, ), name="Decoder_Input"))
    dec.add(tf.keras.layers.Reshape((256, 256, 1), name="Decoder_Reshape_1"))
    dec.add(tf.keras.layers.Conv2D(filters=1, 
                                   kernel_size=(2, 2),
                                   padding="same",
                                   activation="relu", 
                                   data_format="channels_last", 
                                   name="Decoder_Conv2D_1"))
    dec.add(tf.keras.layers.UpSampling2D((2, 2), name="Decoder_UpSampling2D_1"))
    dec.add(tf.keras.layers.Conv2D(filters=4,
                                   kernel_size=(2, 2),
                                   padding="same",
                                   activation="relu",
                                   data_format="channels_last",
                                   name="Decoder_Conv2D_2"))
    dec.add(tf.keras.layers.Conv2D(filters=8,
                                   kernel_size=(4, 4),
                                   padding="same",
                                   activation="relu",
                                   data_format="channels_last", 
                                   name="Decoder_Conv2D_3"))
    dec.add(tf.keras.layers.UpSampling2D((2, 2), name="Decoder_UpSampling2D_2"))
    dec.add(tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(2, 2),
                                   activation="relu",
                                   data_format="channels_last", 
                                   padding="same",
                                   name="Decoder_Conv2D_4"))
    dec.add(tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(1, 1),
                                   activation="sigmoid",
                                   padding="same",
                                   data_format="channels_last",
                                   name="Decoder_Output"))
    return dec

def autoencoder_model():
    ae = tf.keras.models.Sequential(name="AutoEncoder")
    ae.add(tf.keras.layers.Input(shape=(1024, 1024, 1), name="AutoEncoder_Input"))
    ae.add(encoder_model())
    ae.add(decoder_model())
    return ae


def model_summaries():
    encoder_model().summary()
    decoder_model().summary()
    #autoencoder_model().summary()


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
    EPOCHS = 2
    BATCH_SIZE = 16
    model_name = "autoencoder_1_16_ratio2"
    i = idh("d:\\Minor Project\\Dataset\\data")

    #createSubsetDataset(i, lines=1000)
    datasetInfo(i)
    #model_summaries()
    exit(0)

    train_gen = i.datasetGenerator(mode="train", batch_size=BATCH_SIZE, norm=(0, 1), dtype="float32")
    val_gen = i.datasetGenerator(mode="validation", batch_size=BATCH_SIZE, norm=(0, 1), dtype="float32")
    test_gen = i.datasetGenerator(mode="test", batch_size=BATCH_SIZE, norm=(0, 1), dtype="float32")

    train_steps_per_epoch = i.stepsPerEpoch(mode="train", batch_size=BATCH_SIZE)
    val_steps_per_epoch = i.stepsPerEpoch(mode="validation", batch_size=BATCH_SIZE)
    test_steps_per_epoch = i.stepsPerEpoch(mode="test", batch_size=BATCH_SIZE)
    
    #print("STEPS VAL:", val_steps_per_epoch)
    '''
    try:
        os.mkdir("log")
        os.mkdir("log\\scalars")
    except Exception:
        pass

    logdir = "log\\scalars\\"+datetime.now().strftime("%Y%m%d_%H%M%S")
    #using callback is not that imp but chalega
    tensorboard_callback = TensorBoard(log_dir=logdir)
    '''
    ae = autoencoder_model()# lr: 0.07565
    
    ae.compile(optimizer="adam", loss="mse")#, metrics=['mae', 'mse'])

    print("\n\nStarting now\n")
    ae.fit_generator(generator=train_gen, 
                     steps_per_epoch=train_steps_per_epoch,
                     epochs=EPOCHS,
                     verbose=1,
                     validation_data=val_gen,
                     validation_steps=val_steps_per_epoch,
                     use_multiprocessing=False,
                     max_queue_size=5,
                     shuffle=True)
                     #total images in memory = 2*batch_size*max_queue_size; reduce the queue size of batch size to fit the memory if needed
                     
    try:
        ae.save("keras models\\{}.h5".format(model_name))
        model_name = "keras models\\{}.h5".format(model_name)
    except Exception as exx:
        print("\n\nEXCEPTION WHILE SAVING MODEL:", exx)
        ae.save("{}.h5".format(model_name))
        model_name = "{}.h5".format(model_name)

    print("MODEL SAVED SUCCESSFULLY")
    print("\n\n Evaluating now ... \n")
    test_accuracy = ae.evaluate_generator(generator=test_gen, 
                                          steps=test_steps_per_epoch,
                                          use_multiprocessing=False)

    print("\n\n\nTest Accuracy = {}\n".format(test_accuracy))
    return model_name

def check(modelname):
    i = idh("d:\\Minor Project\\Dataset\\data")
    ae = tf.keras.models.load_model(modelname)
    '''
    BATCH_SIZE=16
    test_gen = i.datasetGenerator(mode="test", batch_size=BATCH_SIZE, norm=(0, 1), dtype="float32")
    test_steps_per_epoch = i.stepsPerEpoch(mode="test", batch_size=BATCH_SIZE)
    test_accuracy = ae.evaluate_generator(generator=test_gen, 
                                           steps=test_steps_per_epoch,
                                           use_multiprocessing=False)

    print("\n\n\nTest Accuracy = {}\n".format(test_accuracy))
    '''
    images = i.getImages(df=i.getChunkOfFile(start_index=0, end_index=10, mode="test"), norm=(0, 1), dtype="float32")
    for i, image in enumerate(images):
        print("decoded image shape:", ae.predict(image.reshape(1, 1024, 1024, 1)).shape)
        img = ae.predict(image.reshape(1, 1024, 1024, 1))[0] * 255
        enc_img = ae.layers[0](image.reshape(1, 1024, 1024, 1)).numpy().reshape((256, 256)) * 255
        enc_img = enc_img.astype('uint8')
        img = img.astype('uint8')
        image = image*255
        image = image.astype('uint8')
        cv2.imshow("Encoded_Image", enc_img)
        cv2.imshow("Decoded_Image", img)
        cv2.imshow("Original_Image", image)
        #cv2.imwrite("Test_PredImage_"+str(i)+".png", img)
        #cv2.imwrite("Test_OrigImage_"+str(i)+".png", image)
        #cv2.imwrite("Test_EncodedImage_"+str(i)+".png", enc_img)
        if(cv2.waitKey(0)==ord('q')):
            break


if __name__ == "__main__":
    print("tf version:", tf.__version__) #2.0.0
    print("Keras version:", tf.keras.__version__, "\n") #2.2.4-tf
    #model_summaries()
    modelname = main()
    print("Model Path:", modelname)
    check(modelname)
