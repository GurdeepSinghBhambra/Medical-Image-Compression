import tensorflow as tf
from PIL import Image
import numpy as np
import zipfile
import io


##########
# Models #
##########
AUTOENCODER_MODEL_PATH = "model/autoencoder_1_16_ratio.h5"
Autoencoder = tf.keras.models.load_model(AUTOENCODER_MODEL_PATH, compile=False)
Encoder = Autoencoder.get_layer("Encoder")
Decoder = Autoencoder.get_layer("Decoder")


######################
# ZIPFILE EXCEPTIONS #
######################
class IncorrectFileFormat(Exception):
    pass


class IncorrectImageSize(Exception):
    pass


class IncorrectImageChannels(Exception):
    pass


class EmptyZipFileError(Exception):
    pass


class ZipFileHandler:
    ENCODE_SIZE = (256, 256)
    DECODE_SIZE = (1024, 1024)

    def __init__(self, file_obj):
        self.zipfile = zipfile.ZipFile(file_obj, mode='r')
        self.checkZipFile()

    def checkZipFile(self):
        if(int(len(self.zipfile.infolist())) == 0):
            raise EmptyZipFileError("Zipfile Error: The zipfile is empty")
        for file in self.zipfile.infolist():
            if(".png" not in file.filename):
                raise IncorrectFileFormat("Zipfile Error: Incorrect file type in file \'{}\'. Expects only PNG files.".format(file.filename))
            img = Image.open(io.BytesIO(self.zipfile.read(file)))
            if(img.size not in (ZipFileHandler.ENCODE_SIZE, ZipFileHandler.DECODE_SIZE)):
                raise IncorrectImageSize("Zipfile Error: Incorrect image size {} in file \'{}\'".format(str(img.size), file.filename))
            elif(img.mode != 'L'):
                raise IncorrectImageChannels("Zipfile Error: Incorrect image channels {} found in file {}.")
        return True

    @staticmethod
    def preprocessor(img):
        img = img/255
        return img.astype("float32")

    def processImage(self, img):
        global Encoder, Decoder
        if(img.shape[0] == 1024): #Encode
            img = Encoder.predict(self.preprocessor(img.reshape((1, 1024, 1024, 1)))).reshape((256, 256))*255
        else: #Decode
            img = Decoder.predict(self.preprocessor(img.reshape(1, 256*256))).reshape((1024, 1024))*255
        return Image.fromarray(img.astype('uint8'))

    def autoEncodeDecode(self):
        response_zipfile = io.BytesIO()
        with zipfile.ZipFile(response_zipfile, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.filename = "Processed.zip"
            for file in self.zipfile.infolist():
                img = self.processImage(np.array(Image.open(io.BytesIO(self.zipfile.read(file)))))
                img_file = io.BytesIO()
                img.save(img_file, "PNG")
                zf.writestr(file.filename, img_file.getvalue())
        return response_zipfile.getvalue()


if __name__ == "__main__":
    import time
    st = time.time()
    try:
        print("Model Size Now")
        #time.sleep(10)
        z1 = ZipFileHandler("D:\\Minor Project\\sample_user_file_3.zip")
        #z2 = ZipFileHandler("D:\\Minor Project\\sample_user_file_2.zip")
        #z4 = ZipFileHandler("D:\\Minor Project\\sample_user_file_4.zip")
        
        print("Model+Zipfile")
        #time.sleep(10)
        z1 = z1.autoEncodeDecode()
        #z2 = z2.autoEncodeDecode()
        #z4 = z4.autoEncodeDecode()
        
        #print("Model+Zipfile+new Zipfile")
        #time.sleep(10)
        #print(len(z1), len(z2), len(z4))
        #z = zipfile.ZipFile(io.BytesIO(z), mode='r')
        #print("Model+Zipfile+new Zipfile+new read zipfile")
        #time.sleep(10)
        #print(list(map(lambda x: x.filename, z.infolist())))
        #Image.open(io.BytesIO(z.read(z.infolist()[1]))).show()
    except Exception as exx:
        print(exx)
    print(time.time()-st, "secs")
    time.sleep(10)
