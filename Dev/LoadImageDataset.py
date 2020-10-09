__author__ = "Gurdeep"

exit_flag = False

try:
    from sklearn.model_selection import train_test_split
except Exception:
    print("\nIMPORT ERROR in LoadImageDataset:\n\tsklearn module not present, use \"conda install scikit-learn\" at the anaconda prompt terminal\n")
    exit_flag=True

try:
    from tqdm import tqdm
except Exception:
    print("\nIMPORT ERROR in LoadImageDataset:\n\ttqdm module not present, use \"conda install tqdm\" at the anaconda prompt terminal\n")
    exit_flag = True

try:
    import pandas as pd
except Exception:
    print("\nIMPORT ERROR in LoadImageDataset:\n\tpandas module not present, use \"conda install pandas\" at the anaconda prompt terminal\n")
    exit_flag = True

try:
    from PIL import Image
except Exception:
    print("\nIMPORT ERROR in LoadImageDataset:\n\tpillow module not present, use \"conda install pillow\" at the anaconda prompt terminal\n")
    exit_flag = True

try:
    import numpy as np
except Exception:
    print("\nIMPORT ERROR in LoadImageDataset:\n\tnumpy module not present, use \"conda install numpy\" at the anaconda prompt terminal\n")
    exit_flag = True

try:
    import cv2
except Exception:
    print("\nIMPORT ERROR in LoadImageDataset:\n\topencv module not present, use \"conda install opencv\" at the anaconda prompt terminal\n")
    exit_flag = True

from pathlib import Path
import os

if(exit_flag == True):
    exit(0)

#np.random.seed(42)
 

class ImageDatasetHandler:

    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
        self.data_entry_file_path = None #"Image Index" in file should be sorted. This file has all the dataset info and image path, comes from the zipped folder
        self.total_images = None
        self.checkDir()
        self.data_dir_path = str(Path(data_dir_path).resolve())
        self.image_dir_list = list(map(lambda d_name: str(Path(self.data_dir_path+"/"+d_name+"/images").resolve()), 
                                   ['images_001', 'images_002', 'images_003', 
                                    'images_004', 'images_005', 'images_006', 
                                    'images_007', 'images_008', 'images_009',
                                    'images_010', 'images_011', 'images_012'
                                    ])) # this list must be sorted always
        self.prepared_data_dir = "PreparedData"
        self.makeDir()
        self.master_csv_file = str(Path(self.prepared_data_dir+"/"+"DatasetMasterCsvFile.csv").resolve())
        self.train_csv_file = str(Path(self.prepared_data_dir+"/"+"DatasetTrainCsvFile.csv").resolve())
        self.test_csv_file = str(Path(self.prepared_data_dir+"/"+"DatasetTestCsvFile.csv").resolve())
        self.validation_csv_file = str(Path(self.prepared_data_dir+"/"+"DatasetValidationCsvFile.csv").resolve())

    def makeDir(self):
        try:
            os.mkdir(self.prepared_data_dir)
            print("\n\"PreparedData\" Directory created in current directory\n")
            self.prepared_data_dir = str(Path(self.prepared_data_dir).resolve())
        except Exception:     
            pass

    def checkDir(self):
        if(os.path.isdir(self.data_dir_path) == False):
            print("ERROR in LoadImageDataset/checkDir:\n\t\'{}\' data directory path incorrect\n".format(self.data_dir_path))
            exit(0)

        data_dir_list = os.listdir(self.data_dir_path)

        if("Data_Entry_2017.csv" not in data_dir_list):
            print("\nERROR in LoadImageDataset/checkDir:\n\t\'{}\' file not present\n".format("Data_Entry_2017.csv"))
            exit(0)
        self.data_entry_file_path = str(Path(self.data_dir_path+"/"+"Data_Entry_2017.csv").resolve())
        
        image_dir_list = ['images_001', 'images_002', 'images_003', 
                          'images_004', 'images_005', 'images_006', 
                          'images_007', 'images_008', 'images_009',
                          'images_010', 'images_011', 'images_012']
        
        total_images = 0
        for image_dir in image_dir_list:
            if(image_dir not in data_dir_list):
                print("\nERROR in LoadImageDataset/checkDir:\n\t\'{}\' image dir missing in data dir, contact Gurdeep abhi ke abhi\n".format(image_dir))
                exit(0)
            total_images += int(len(os.listdir(self.data_dir_path+"/"+image_dir+"/images")))
        self.total_images = total_images

        if("BBox_List_2017.csv" not in data_dir_list):
            print("\nWARNING in LoadImageDataset/checkDir:\n\t\'{}\' file not present\n".format("BBox_List_2017.csv"))
        
    def displayDirReport(self):
        print("\nMaster Directory Summary:")
        print("\tTotal Image Dirs: {}".format(len(self.image_dir_list)))
        for dir_no, image_dir in enumerate(self.image_dir_list, 1):
            dir_image_count = len(os.listdir(image_dir))
            print("\n\t\t-----------------------------------------------------------")
            print("\t\tdir_path:", image_dir)
            print("\t\tNo of Images:", dir_image_count)
            print("\n\t\t-----------------------------------------------------------")
        print("\tTotal Images:", self.total_images)
        print("\n")
    
    def openFile(self, mode):
        if(mode.lower() == 'master'):
            if(os.path.isfile(self.master_csv_file)):
                return pd.read_csv(self.master_csv_file, index_col=0)
            else:
                print("\nERROR in LoadImageDataset/openFile:\n\tmaster csv file does not exists.\n")
        elif(mode.lower() == 'train'):
            if(os.path.isfile(self.train_csv_file)):
                return pd.read_csv(self.train_csv_file, index_col=0)
            else:
                print("\nERROR in LoadImageDataset/openFile:\n\ttrain csv file does not exists.\n")
        elif(mode.lower() == 'test'):
            if(os.path.isfile(self.test_csv_file)):
                return pd.read_csv(self.test_csv_file, index_col=0)
            else:
                print("\nERROR in LoadImageDataset/openFile:\n\ttest csv file does not exists.\n")
        elif(mode.lower() == 'validation'):
            if(os.path.isfile(self.validation_csv_file)):
                return pd.read_csv(self.validation_csv_file, index_col=0)
            else:
                print("\nERROR in LoadImageDataset/openFile:\n\tvalidation csv file does not exists.\n")
        else:
            print("\nERROR in LoadImageDataset/saveFile:\n\tWrong/incompatible mode type given\n")
        exit(0)

    def saveFile(self, df, mode):
        if(mode.lower() == 'master'):
            df.to_csv(self.master_csv_file)
            self.master_csv_file = str(Path(self.master_csv_file).resolve())
        elif(mode.lower() == 'train'):
            df.to_csv(self.train_csv_file)
            self.train_csv_file = str(Path(self.train_csv_file).resolve())
        elif(mode.lower() == 'test'):
            df.to_csv(self.test_csv_file)
            self.test_csv_file = str(Path(self.test_csv_file).resolve())
        elif(mode.lower() == 'validation'):
            df.to_csv(self.validation_csv_file)
            self.validation_csv_file = str(Path(self.validation_csv_file).resolve())
        else:
            print("\nERROR in LoadImageDataset/saveFile:\n\tWrong/incompatible mode type given\n")
            return False
        return True

    def getFileRows(self, mode, df=None):
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("\nERROR in LoadImageDataset/getFileRows:\n\tNo DataFrame given, it expects it because mode is None/not string\n")
                exit(0) 
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("\nERROR in LoadImageDataset/getFileRows:\n\tWrong/incompatible mode given\n")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        return int(len(df.index))

    def deleteFileRows(self, mode, rows_to_delete, print_it=False, df=None):
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("\nERROR in LoadImageDataset/getFileRows:\n\tNo DataFrame given, it expects it because mode is None/not string\n")
                exit(0) 
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("\nERROR in LoadImageDataset/getFileRows:\n\tWrong/incompatible mode given\n")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        if(rows_to_delete <= 0):
            return df

        drop_indexes = np.random.choice(df.index, rows_to_delete, replace=False)
        if(print_it == True):
            print("\nDelete Summary:\n")
            print("Deleting Index:", drop_indexes)
            print("Rows to be deleted")
            print(df.iloc[drop_indexes])
            print("\n")
        df.drop(drop_indexes, inplace=True)
        df.reset_index(inplace=True)
        df.drop(columns=['index'], inplace=True)

        return df

    def createMasterDataframe(self, skip=False, clean=False):
        print("\nCreating Master DataFrame, This might take some time. Sit back, relax :)\n")
        try:
            df = pd.read_csv(self.data_entry_file_path)
        except Exception as exx:
            print("\nERROR in LoadImageDataset/createMasterDataframe:\n\t{}\n".format(exx))
            exit(0)

        master_df = df.copy()
        column_rename_dict = {'Image Index':'Image_Path',
                              'Finding Labels':'Finding_Labels',
                              'Follow-up #':'Follow_Ups',
                              'Patient ID':'Patient_Id',
                              'Patient Age':'Patient_Age',
                              'Patient Gender':'Patient_Gender',
                              'View Position':'View_Position',
                              'OriginalImage[Width':'Original_Image_Width',
                              'Height]':'Original_Image_Height',
                              'OriginalImagePixelSpacing[x':'Original_Pixel_Spacing_X',
                              'y]': 'Original_Pixel_Spacing_Y',
                              }
        unwanted_columns = list(set(df.keys())-set(column_rename_dict.keys()))
        master_df.drop(columns=unwanted_columns, axis=1, inplace=True)
        master_df.rename(columns=column_rename_dict, inplace=True)

        if(clean == True):
            if(master_df.isnull().values.any()==True):
                print("\nWARNING in LoadImageDataset/createMasterDataframe:\n\tCleaning/Deleting {} Null/NaN/NaT rows\n".format(master_df.isnull().sum().sum()))
                master_df.dropna(axis=0, inplace=True)
                master_df.reset_index(inplace=True)
                master_df.drop(columns=['index'], inplace=True)

        master_df['Image_Shape'] = pd.Series(np.empty(int(len(master_df.index)), dtype="<U"), index=master_df.index)

        image_dir_iter, skipped_files = 0, []
        curr_image_dir_list = os.listdir(self.image_dir_list[image_dir_iter])
        for i in tqdm(range(int(len(master_df.Image_Path.values))), smoothing=0.001):
            image_filename = master_df.Image_Path[i]
            break_flag=False
            while(image_filename not in curr_image_dir_list):
                if((image_dir_iter+1) < int(len(self.image_dir_list))):
                    image_dir_iter = image_dir_iter+1
                    curr_image_dir_list = os.listdir(self.image_dir_list[image_dir_iter])
                else:
                    if(skip == True):
                        skipped_files.append(image_filename)
                        image_dir_iter = 0
                        curr_image_dir_list = os.listdir(self.image_dir_list[image_dir_iter])
                        break_flag = True
                    else:
                        print("\nERROR in LoadImageDataset/createMasterDataframe:\n\t{} file not present in data_entry file\n\tif you want to skip, use 'skip=True' while calling this function")
                        exit(0)
                if(break_flag == True):
                    break
            if(break_flag == True):
                continue
            image_path = str(Path(self.image_dir_list[image_dir_iter]+"/"+image_filename).resolve())
            master_df.at[i, "Image_Path"] = image_path
            master_df.at[i, "Image_Shape"] = str(np.array(Image.open(image_path)).shape)[1:-1]    

        if(skip == True and int(len(skipped_files)) > 0):
            print("\nWARNING in LoadImageDataset/createMasterDataframe:\n\t{} file(s) skipped\n".format(len(skipped_files)))

        master_df.drop(master_df[master_df.Image_Shape == "1024, 1024, 4"].index, inplace=True)
        master_df.reset_index(inplace=True)
        master_df.drop(columns=['index'], inplace=True)
        print("\n")

        return master_df

    def getChunkOfFile(self, start_index, end_index, mode, df=None):
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("\n\tERROR in LoadImageDataset/getChunkOfFile: No DataFrame given, it expects it because mode is None/not string\n")
                exit(0) 
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("\nERROR in LoadImageDataset/getChunkOfFile:\n\tWrong/incompatible mode given\n")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        if(start_index < 0 or start_index > int(len(df.index))):
            print("\nError in LoadImageDataset/getChunkOfFile:\n\tstart_index out of bound\n")
            exit(0)
        if(end_index < 0 or end_index > int(len(df.index))):
            print("\nError in LoadImageDataset/getChunkOfFile:\n\tend_index out of bound\n")
            exit(0)
        return df.iloc[start_index:end_index]

    @staticmethod
    def normalize(nparr, a, b, preserve_dtype=True):
        try:
            if(preserve_dtype):
                if('int' in str(nparr.dtype)):
                    return a + (nparr * ((b-a)//nparr.max()))
                elif('float' in str(nparr.dtype)):
                    return a + (nparr * ((b-a)/nparr.max()))
                else:
                    raise Exception("Invalid Numpy Array Data-type, supported dtypes are \'float\' and \'int\'")
            else:
                return a + (nparr * ((b-a)/nparr.max()))
        except Exception as exx:
            print("ERROR in LoadImageDataset/normalize:", exx)
            exit(0)

    @staticmethod
    def crop_image(image, y, yh, x, xh):
        return image[y:y+yh, x:x+xh]

    def getImages(self, df, flatten=False, reshape_for_keras=True, dtype=None, norm=None, crop=None):        
        if(crop!=None):
            if(type(crop) not in [type(list()), type(tuple())]):
                print("ERROR in LoadImageDataset/getImages: crop should be a tuple/list")
                exit(0)
            if(int(len(crop)) != 4):
                print("ERROR in LoadImageDataset/getImages: crop tuple/list should be of size 4\n\tCurrent size is", len(norm))
                exit(0)
            if(flatten==True):
                print("ERROR in LoadImageDataset/getImages: Flatten and crop cannot be used together, sury :(")
                exit(0)

        images, shape, do_crop = None, None, crop!=None
        for i, image_path in enumerate(df.Image_Path, 0):
            img = np.array(Image.open(image_path))
            if(flatten == True):
                img = img.flatten()
            if(i==0):
                shape = [int(len(df.index))]
                if(do_crop):
                    shape.extend(self.crop_image(img, crop[0], crop[1], crop[2], crop[3]).shape)
                else:
                    shape.extend(img.shape)
                images = np.empty(shape, dtype=img.dtype)
            if(do_crop):
                images[i] = self.crop_image(img, crop[0], crop[1], crop[2], crop[3])
            else:
                images[i] = img

        if(reshape_for_keras == True):
            shape = list(images.shape)
            shape.append(1)
            images = images.reshape(shape)

        if(norm!=None):
            if(type(norm) not in [type(list()), type(tuple())]):
                print("ERROR in LoadImageDataset/getImages: norm (for normalization) should be a tuple/list")
                exit(0)
            if(int(len(norm)) != 2):
                print("ERROR in LoadImageDataset/getImages: norm (for normalization) tuple/list should be of size 2\n\tCurrent size is", len(norm))
                exit(0)
            if(norm[0] > norm[1]):
                print("ERROR in LoadImageDataset/getImages: norm (for normalization) tuple/list must be in ascending order")
                exit(0)
            images = self.normalize(nparr=images, a=norm[0], b=norm[1], preserve_dtype=False)

        if(dtype != None):
            images = images.astype(dtype)

        return images

    @staticmethod
    def showImages(image_array, window_refresh_delay=500):
        if(image_array.ndim<=2):
            print("\nERROR in LoadImageDataset/showImages:\n\tArray Dimension Incorrect, Expecting array of dimension > 2\n")
            exit(0)
        if(window_refresh_delay<0):
            print("ERROR in LoadImageDataset/showImages:\n\twindow_refresh_delay cannot be less than 0\n")
            exit(0)
        elif(window_refresh_delay==0):
            print("WARNING in LoadImageDataset/showImages:\n\twindow_refresh_delay is 0, press any key to change image, 'q' for quit.\n")

        for img in image_array:
            cv2.imshow("Images", img)
            if(cv2.waitKey(window_refresh_delay) == ord('q')):
                break
        cv2.destroyAllWindows()

    def train_test_split(self, df=None, ret=False, validation=False, test_size=0.15, validation_size=0.15, random_state=None, shuffle=False):
        if(type(df) == type(None) and df == None):
            df = self.openFile(mode='master')

        X, y = df.drop(['Image_Path'], axis=1), df.drop([col for col in df.keys() if col != 'Image_Path'], axis=1)

        total_size = int(len(y.Image_Path))

        train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=None, test_size=test_size, random_state=random_state, shuffle=shuffle)

        print("\nSplit Summary:")

        if(validation == True):
            val_size = (validation_size*total_size)/int(len(train_y.Image_Path))
            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=None, test_size=val_size, random_state=random_state, shuffle=shuffle)
            validation_df = pd.concat([val_x, val_y], axis=1)
            validation_df.reset_index(drop=True, inplace=True)
            self.saveFile(validation_df, mode='validation')
            print("validation size =", len(validation_df.Image_Path))
            print("percent of validation size, wrt to master file =", (int(len(validation_df.Image_Path))/total_size)*100, "%")

        train_df = pd.concat([train_x, train_y], axis=1)
        train_df.reset_index(drop=True, inplace=True) 
        test_df = pd.concat([test_x, test_y], axis=1)
        test_df.reset_index(drop=True, inplace=True)
        self.saveFile(train_df, mode='train')
        self.saveFile(test_df, mode='test')

        print("given test and validation sizes =", test_size, validation_size)
        print("test size =", len(test_df.Image_Path))
        print("percent of test size, wrt to master file =", (int(len(test_df.Image_Path))/total_size)*100, "%")
        print("train size =", len(train_df.Image_Path))
        print("percent of train size, wrt to master file =", (int(len(train_df.Image_Path))/total_size)*100, "%")
        
        if(ret == False):
            return None
        elif(validation == True):
            return train_x, test_x, val_x, train_y, test_y, val_y   
    
        return train_x, test_x, train_y, test_y

    @staticmethod
    def getFactors(no):
        return [n for n in range(1, no+1) if(no%n == 0)]

    def getBatchSizes(self, modes=['train', 'test', 'validation', 'master'], df=None):
        batch_size_dict = dict()
        if(df is None):
            for file_mode in modes:
                batch_size_dict[file_mode] = self.getFactors(self.getFileRows(mode=file_mode))
        else:
            batch_size_dict['Unknown File'] = self.getFactors(int(len(df.index)))
        return batch_size_dict

    def shapeAccordingToBatchSize(self, batch_size, mode, save=False, df=None, print_it=True):
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("\nERROR in LoadImageDataset/shapeAccordingToBatchSize:\n\tNo DataFrame given, it expects it because mode is None/not string\n")
                exit(0)
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("\nERROR in LoadImageDataset/shapeAccordingToBatchSize:\n\tWrong/incompatible mode given\n")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        if(print_it == True):
            print("\nTotal rows deleted to match the batch size =", int(len(df.index))%batch_size, "\n")

        df = self.deleteFileRows(mode=None, df=df, rows_to_delete=int(len(df.index))%batch_size)

        if(save == True):
            if(mode == None):
                print("\nERROR in LoadImageDataset/shapeAccordingToBatchSize:\n\tmode is None, cannot save file\n")
                exit(0)
            self.saveFile(df, mode=mode)
        return df

    def stepsPerEpoch(self, mode, batch_size, df=None):
        if(df is not None):
            df = self.shapeAccordingToBatchSize(mode=None, df=df, batch_size=batch_size, print_it=False)
        else:
            df = self.shapeAccordingToBatchSize(mode=mode, batch_size=batch_size, print_it=False)
        steps_per_epoch = self.getFileRows(mode=None, df=df)/batch_size
        return steps_per_epoch

    def datasetGenerator(self, batch_size, mode, df=None, flatten=False, reshape_for_keras=True, dtype=None, norm=None, crop=None):
        file_iter =  0
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("\nERROR in LoadImageDataset/datasetGenerator:\n\tNo DataFrame given, it expects it because mode is None/not string\n")
                exit(0)
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("\nERROR in LoadImageDataset/datasetGenerator:\n\tWrong/incompatible mode given\n")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        df = self.shapeAccordingToBatchSize(batch_size=batch_size, mode=None, df=df)

        df_max_size = self.getFileRows(mode=None, df=df)

        while(True):
            if(file_iter+batch_size > df_max_size):
                file_iter = 0
                continue

            chunked_df = self.getChunkOfFile(start_index=file_iter, end_index=file_iter+batch_size, mode=None, df=df)
            images = self.getImages(df=chunked_df, flatten=flatten, reshape_for_keras=reshape_for_keras, dtype=dtype, norm=norm, crop=crop)

            yield (images, images)

            file_iter+=batch_size


def test():
    d = ImageDatasetHandler("d:\\Minor Project\\Dataset\\data")
    #print(d.data_dir_path)
    #print(d.image_dir_list)
    #print(d.data_entry_file_path)
    #d.displayDirReport()
    #df = d.createMasterDataframe(clean=True)
    #df = d.openFile(mode='master')
    #shapes = df.Image_Shape.unique()
    #print(shapes)
    #print("Shape Selected =", shapes[1])
    #print("Shape Count =", len(df[df.Image_Shape == "1024, 1024, 4"]))
    #df.drop(df[df.Image_Shape == "1024, 1024, 4"].index, inplace=True)
    #df.reset_index(inplace=True)
    #df.drop(columns=['index'], inplace=True)
    #df.info()
    #print(len(df.loc[df.Image_Shape == "1024, 1024, 4", "Image_Shape"]))
    #print(df.Image_Shape.unique())
    #print(d.getImageDatasetShapes(df=df))
    #print(d.getChunkOfFile(mode=None, df=df, start_index=int(len(df.index))-1, end_index=int(len(df.index))))
    #center crop for 1024, 1024 images
    crop = (256, 512, 256, 512)
    #images = d.getImages(df=d.getChunkOfFile(mode='validation', start_index=0, end_index=10), flatten=False, dtype='float32', norm=(0, 1), crop=crop)#, reshape_for_keras=True)
    #print(images[0][0][0])
    #print(images.ndim)
    #print(images.shape)
    #d.showImages(image_array=images, window_refresh_delay=0)
    #print(df.info())
    #print("\n")
    #print(df.iloc[10:20])
    #d.saveFile(df=df, mode='master')
    #df.info()
    #print(df)
    #print(df.isnull().sum().sum())
    #df = d.deleteFileRows(mode='master', rows_to_delete=20)
    #df.info()
    #print(d.shapeAccordingToBatchSize(batch_size=32, mode='test').info())
    #d.train_test_split(validation=True, validation_size=0.25)
    #print(d.getBatchSizes())
    #df = d.openFile(mode='train')
    #print("steps_per_epoch:", d.stepsPerEpoch(mode='train', batch_size=64))
    for i, image in enumerate(d.datasetGenerator(mode='validation', batch_size=50, norm=(0, 1), dtype='float64', crop=crop)):
        #print(np.all(image[0].astype("float32")/255 == d.normalize(image[0], 0, 1, False).astype("float32")))
        print(image[0][0][0][0])
        print(image[0].shape)
        print(image[0].dtype)
        #break
        d.showImages(image_array=image[0], window_refresh_delay=0)
        if(i==1):
            break

#test()