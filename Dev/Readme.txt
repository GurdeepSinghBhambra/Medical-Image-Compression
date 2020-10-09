Dataset:
	Unzip the zip folder "data.zip" (its the zipped dataset file).
	Here, data_directory refers to the directory having the unzipped contents of the dataset.
	Make sure all the content is untouched after its extracted and DO NOT CHANGE ANYTHING IN THE DIRECTORY.

	Directory Contents:
		
		data--
		      |--Images_001--
		      |              |--images--
		      | 		        |-- "*.png"
		      |
		      |--Images_002--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_003--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_004--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_005--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_006--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_007--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_008--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_009--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_010--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_011--
		      |              |--images--
		      | 			    |-- "*.png"
		      |		      
		      |--Images_012--
		      |              |--images--
		      | 			    |-- "*.png"
		      |
		      |--ARXIV_V5_CHESTXRAY.pdf		      
		      |
		      |--BBox_List_2017.csv (REQUIRED, IF NOT PRESENT WARNING WILL BE PRINTED)		      
		      |		      
		      |--Data_Entry_2017.csv (IMP)		      
		      |		      
		      |--FAQ_CHESTXRAY.pdf
		      |		      
		      |--LOG_CHESTXRAY.pdf		      
		      |		      
		      |--README_CHESTXRAY.pdf		      
		      |		      
		      |--test_list.txt
		      |
              |--train_val_list.txt

	Summary:
		Our Dataset has 519 Images of 4 color channels and rest 111601 are single channel images.
		I delete the 519 images while create dataframe as it is not of any practical use.
		The dataset has 12 Image dirs and 2 important csv files. The "Data_Entry_2017.csv" is very 
		important as it is used to create the master dataframe in the library. 
		SO, DO NOT OPEN THE FILE WITH WPS OFFICE AND SAVE OR MAKE ANY TYPE OF CHANGES, IF CHANGED
		PROGRAM WILL GET FUCKED UP.

Use the LoadImageDataset Library for extraction, pre-processing of the dataset

Dependencies of LoadImageDataset are:
	sklearn, opencv, pillow, tqdm, numpy, pandas

if you want to install any of the dependent libraries, follow the following steps:
	1. Press "windows_key + s" and type "anaconda prompt" and launch it.
	2. Activate your conda environment by typing "conda activate YOUR_ENVIRONMENT_NAME". if you dont remeber, then:
		2.1 Type "conda env list" and press enter.
		2.2 now see the environment name and follow step 2.
	3. Now with your environment activated, type "conda install PACKAGE_NAME" and press "y" when prompted during the installation.
	4. You have successfully installed the libraries.

LoadImageDataset Versions:
	v1.0 (Launch)
    v1.1 (minor changes and new feature)
    v1.2 (minor changes and new micro-features)

Docs of LoadImageDataset are mentioned below:
    
    LoadImageDataset has following classes:

    Classes:
            ImageDatasetHandler: This class will extract data, process data and comes with a generator to train, test, validate
                                 all neural networks. Make sure to unzip the 'data.zip' folder and save it in a directory. That 
                                 unzipped folder will be the input to this class's objects. Any mistake or missing file will throw
                                 a exception and terminate execution.
                                 Note: When an object of this class is created, it also creates a directory named "PreparedData" which
                                       houses all the important created dataframes like master, test, validation, train.
                                 Warning: if you specify 2 different dataset directories, all the data goes into the same "PreparedData"
                                          folder. All the files in the Directory will be overwritten each time you save any file, using 
                                          the saveFile function  

    Pointers:
            -> Here mode takes 5 values: 'test', 'train', 'validation', 'master' and None. Read Each function's mode inputs, see the accepted mode types
                                       and then use the appropriate values.
                                       'master' means master csv file or dataframe that contains whole the dataset info.
                                       'train' means train csv file or dataframe that contains the dataset info for training.
                                       'test' means test csv file or dataframe that contains the dataset info for testing.
                                       'validation' means validation csv file or dataframe that contains the dataset info for validating.
                                       Whichever function has mode=None as valid value, then make sure you provide your own dataframe.

            -> Whenever you make changes or use your own processed dataframe, make sure to not change or alter the format of the files, other wise
               abnormal termination will occur.

            -> Make sure to read the function's default values for its arguments specified in its syntax below

        
    Functions in ImageDatsetHandler (and if you have any doubts in using them, See Examples):
            displayDirReport():
                            input: None
                            returns: None
                            Description: "Displays info about the data directory"

            openFile(mode):
                            inputs: mode: specify with file to open. (accepted modes: 'train', 'test', 'master', 'validation')
                            returns: Pandas dataframe 
                            Description: "opens a csv file and returns pandas dataframe"

            saveFile(df, mode):
                            inputs: mode: specify the file to save to. (accepted modes: 'train', 'test', 'master', 'validation')
                                    df: pandas dataframe (in correct format)    
                            returns: True if successful else False
                            Description: "saves pandas dataframe to csv file"

            getFileRows(mode, df=None):
                            inputs: mode: specify the file mode. (accepted modes: 'train', 'test', 'master', 'validation' or None)
                                    df: pandas dataframe (in correct format)
                            returns: integer (total rows in the file)
                            Description: "gets the no of rows of a file"
                            Note: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                  provide your own dataframe with df=YOUR_DATAFRAME

            deleteFileRows(mode, rows_to_delete, print_it=False, df=None):
                            inputs: mode: specify the file mode. (accepted modes: 'train', 'test', 'master', 'validation' or None)
                                    row_to_delete: Integer specifying no of rows to delete
                                    print_it: Bool variable to print the delete summary. Set True to print the summary.
                                    df: pandas dataframe (in correct format)
                            returns: New trimmed pandas dataframe    
                            Description: "Deletes the rows of the specified file"
                            Note: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                  provide your own dataframe with df=YOUR_DATAFRAME

            createMasterDataframe(skip=False, clean=False):
                            inputs: skip: Set true to skip any file which is not present or invokes any type of exception
                                    clean: Set True to clean all the NaN,
                                     NaT, Null values, if any.
                            returns: Pandas dataframe
                            Description: "This is the main Function to create the master dataframe which handles the whole dataset"
                            Warning: You have to use this function once (since it takes about 25 to 30 mins to complete) and save the 
                                     dataframe using 'saveFile()' function. 
                            Note: This is the main function this library requires so make sure you run it once and save it. (See Examples)

            getChunkOfFile(start_index, end_index, mode, df=None):
                            inputs: start_index: starting index to extract rows/chunk of dataframe/csv file
                                    end_index: last index to extract rows/chunk of dataframe/csv file
                                    mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
                                    df: pandas dataframe (in correct format)
                            returns: pandas dataframe (sliced dataframe for given original dataframe)
                            Description: "Slices the given dataframe as per the start_index and end_index"
                            Note: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                  provide your own dataframe with df=YOUR_DATAFRAME

  (updated) getImages(df, flatten=False, reshape_for_keras=True, dtype=None, norm=None):
                            inputs: df: pandas dataframe (in correct format)
                                    flatten: boolean value specifying to convert original image to 1-D image
                                    reshape_for_keras: boolean value specifying to shape input for neural networks
                                    dtype: data type of the final images (this applies at at last)
                                    norm: A python tuple having values for image normalization in ascending order 
                            return: Numpy array of images (with no of images = no of rows in given df)
                            Description: "This Function extracts images from the given dataframe"

            showImages(image_array, window_refresh_delay=500):
                            inputs: image_array: Numpy array having image arrays (like output of getImages)
                                    window_refresh_delay: Integer no between 0 - infinity which specifies the interval (in ms) between each image
                            returns: None
                            Description: "Use this function to see the images"
                            Note: When the window will open, if you want to exit while its running, press 'q' key.
                            Warning: If the window_refersh_delay is 0, then press any key to display next image. Upon pressing 'q' key, the display window 
                                     will close.

            train_test_split(df=None, ret=False, validation=False, test_size=0.15, validation_size=0.15, random_state=None, shuffle=False):
                            inputs: df: pandas dataframe (in correct format)
                                    ret: boolean value specifying whether to return the split dataset arrays or not.
                                    validation: boolean value specifying whether to split the dataset into 2 sets (train, test) or 3 sets(train, set, validation)
                                    test_size: float value specifying the size of test dataset wrt master dataset (master dataset is master dataframe)
                                    validation_size: float value specifying the size of validation dataset wrt master dataset (master dataset is master dataframe)
                                    random_state: same as the random_state in sklearn train_test_split
                                    shuffle: boolean value specifying whether to shuffle or not (same as the shuffle in sklearn train_test_split)
                            returns: if ret true: returns train_x, test_x, train_y, test_y or train_x, test_x, val_x, train_y, test_y, val_y (based on validation)
                                     else: None
                            Description: "Splits the dataset into train, test, validation sub sets and also saves them in PreparedData directory"
                            Notes: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                   provide your own dataframe with df=YOUR_DATAFRAME. 
                                   Here, y (target variable) is Image_Path column in the dataframe since splits are done according to it, just for splitting, 
                                   doesn't mean its the target variable.

            getBatchSizes(files=['train', 'test', 'validation', 'master'], df=None):
                            inputs: files: python list specifying the file modes
                                    df: pandas dataframe (in correct format)
                            returns: python dictionary having file type and its possible batch sizes
                            Description: "Gives file type and its possible batch sizes without deleting any row in the file"
                            Note: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                  provide your own dataframe with df=YOUR_DATAFRAME

            shapeAccordingToBatchSize(batch_size, mode, save=False, df=None):
                            inputs: batch_size: batch_size for the dataset
                                    mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
                                    save: boolean value specifying whether to save the new trimmed file or not. Does not work for mode = None
                                    df: pandas dataframe (in correct format)
                            returns: trimmed pandas dataframe
                            Description: "Trims the Dataframe according to batch_size"
                            Note: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                  provide your own dataframe with df=YOUR_DATAFRAME

  (updated) datasetGenerator(batch_size, mode, df=None, flatten=False, reshape_for_keras=True, dtype=None, norm=None):
                            inputs: batch_size: batch_size for the dataset
                                    mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
                                    df: pandas dataframe (in correct format)
                                    flatten: boolean value specifying to convert original image to 1-D image
                                    reshape_for_keras: boolean value specifying to shape input for neural networks (for 2d images instead of width x height 
                                                       there will be width x height x 1)
                                    dtype: data type of the final images (this applies at at last)
                                    norm: a python tuple having values for image normalization in ascending order 
                            returns: python generator object
                            Description: "This is a infinite loop Python generator for iterating over the dataset in batches. Use this function for 
                                          keras model.fit_generator"
                            Warning: Make sure you pay attention to any error, warnings or statements produced by this function
                            Note: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                  provide your own dataframe with df=YOUR_DATAFRAME

      (new) stepsPerEpoch(mode, batch_size, df=None):
                            inputs: mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
                                    batch_size: batch_size for the dataset
                                    df: pandas dataframe (in correct format)
                            returns: a float number 
                            Description: "This function returns the no of steps per epoch for the given dataframe that you need for the model.fit_generator"
                            Note: If mode is None, then df must have a dataframe. If you wish to provide your own dataframe then set mode=None and  
                                  provide your own dataframe with df=YOUR_DATAFRAME
                            Warning: When using your own specified dataframe (mode=None, df=YOUR_DATAFRME) make sure the generator is set accordingly, since based
                                  on this number, fit_generator will 

Examples:
Here, my extracted folder is Dataset\data. Use yours accordingly.
Here, each session means i started a new python inter 

--------------------------------------------------------------------------------------------------------------------------------------------------
New Session:


//Import the class from the module as below
>>> from LoadImageDataset import ImageDatasetHandler as idh 


//Create the class object as below
//Only if there is no folder named "PreparedData" in the working directory then only it will create a folder and prompt the same
>>> i = idh('Dataset\\data')

"PreparedData" Directory created in current directory


//No folder is created now since there is one already.
>>> i = idh('Dataset\\data')


//Display the directory images and its report
>>> i.displayDirReport() 
[ Ouput bahat bada hai so, output ke liye try for yourselves ]


//CREATING MASTER DATAFRAME IS VERY IMPORTANT AND IS USED FOR
//ALMOST ALL THE OPERATIONS IN THIS LIBRARY.    
//Create the master dataframe and make sure to save the master dataframe as shown below
>>> master_df = i.createMasterDataframe(clean=True) # clean = True will remove any Null, NaN, NaT values
>>>master_df
                                               Image_Path          Finding_Labels  ...  Original_Pixel_Spacing_Y  Image_Shape
0       D:\Minor Project\Dataset\data\images_001\image...            Cardiomegaly  ...                     0.143   1024, 1024
1       D:\Minor Project\Dataset\data\images_001\image...  Cardiomegaly|Emphysema  ...                     0.143   1024, 1024
2       D:\Minor Project\Dataset\data\images_001\image...   Cardiomegaly|Effusion  ...                     0.168   1024, 1024
3       D:\Minor Project\Dataset\data\images_001\image...              No Finding  ...                     0.171   1024, 1024
4       D:\Minor Project\Dataset\data\images_001\image...                  Hernia  ...                     0.143   1024, 1024
...                                                   ...                     ...  ...                       ...          ...
111596  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024
111597  D:\Minor Project\Dataset\data\images_012\image...          Mass|Pneumonia  ...                     0.168   1024, 1024
111598  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024
111599  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024
111600  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024

[111601 rows x 12 columns]


//Save The master dataframe
>>> i.saveFile(df=master_df, mode='master') #make sure you specify mode='master'
True


//Open any of the train, test, master, validation
//opening train csv file
>>> master_df = i.openFile(mode='master')
>>> master_df
                                              Image_Path          Finding_Labels  ...  Original_Pixel_Spacing_Y  Image_Shape
0       D:\Minor Project\Dataset\data\images_001\image...            Cardiomegaly  ...                     0.143   1024, 1024
1       D:\Minor Project\Dataset\data\images_001\image...  Cardiomegaly|Emphysema  ...                     0.143   1024, 1024
2       D:\Minor Project\Dataset\data\images_001\image...   Cardiomegaly|Effusion  ...                     0.168   1024, 1024
3       D:\Minor Project\Dataset\data\images_001\image...              No Finding  ...                     0.171   1024, 1024
4       D:\Minor Project\Dataset\data\images_001\image...                  Hernia  ...                     0.143   1024, 1024
...                                                   ...                     ...  ...                       ...          ...
111596  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024
111597  D:\Minor Project\Dataset\data\images_012\image...          Mass|Pneumonia  ...                     0.168   1024, 1024
111598  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024
111599  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024
111600  D:\Minor Project\Dataset\data\images_012\image...              No Finding  ...                     0.168   1024, 1024

[111601 rows x 12 columns]


//TO CREATE TRAIN, TEST, VALIDATION PORTIONS FROM THE DATASET USE THIS (IMP)
//Here i have specified the validation=True for creating a validating dataset
//Also i have specified the portion for validation size as 0.25 (meaning 25% of the whole dataset)
//In same way test size is 0.15 (meaning 15% of whole dataset)
//shuffle=True for shuffling the dataset randomly
//Check the output to see the sizes of the created portions
//This function automatically saves the files
>>> i.train_test_split(validation=True, validation_size=0.25, test_size=0.15, shuffle=True)

Split Summary:
validation size = 27901
percent of validation size, wrt to master file = 25.000672036988913 %
given test and validation sizes = 0.15 0.25
test size = 16741
percent of test size, wrt to master file = 15.000761641920771 %
train size = 66959
percent of train size, wrt to master file = 59.99856632109031 %



//If you wish to have only test and train dataset then use like as shown below
//Default Value of validation parameter is False so no need to specify it as False
>>> i.train_test_split(test_size=0.20, shuffle=True)

Split Summary:
given test and validation sizes = 0.2 0.15
test size = 22321
percent of test size, wrt to master file = 20.000716839454842 %
train size = 89280
percent of train size, wrt to master file = 79.99928316054515 %


//If you want the output instead of just saving it, then use it like shown below
//WARNING: Since your are using this make sure 
>>>

--------------------------------------------------------------------------------------------------------------------------------------------------
New Session:


>>> from LoadImageDataset import ImageDatasetHandler as idh
>>> i = idh('Dataset/data')
>>> i.train_test_split(validation=True, validation_size=0.25, test_size=0.15, shuffle=True)

Split Summary:
validation size = 27901
percent of validation size, wrt to master file = 25.000672036988913 %
given test and validation sizes = 0.15 0.25
test size = 16741
percent of test size, wrt to master file = 15.000761641920771 %
train size = 66959
percent of train size, wrt to master file = 59.99856632109031 %

//Get the file size/no of rows in the file (no of rows = no of images)
>>> i.getFileRows(mode='train')
66959

//Get the file size with your own dataframe
>>> my_own_df = i.openFile(mode='master') # here i opened the master file, but you can do any operations and then feed it to the function
>>> i.getFileRows(mode=None, df=my_own_df)
111601

//Delete File rows as shown below
//Size before deletion
>>> i.getFileRows(mode=None, df=i.openFile(mode='validation'))
27901

//After deletion
//rows_to_delete=5, specifying i want to delete 5 rows
//print_it=True, specifying to print the deletion summary having the deleted row. 
>>> trimmed_df = i.deleteFileRows(mode='validation', rows_to_delete=5, print_it=True)
Deleting Index: [17990 10051  1331 25371 10860]
Rows to be deleted
             Finding_Labels  Follow_Ups  ...  Image_Shape                                         Image_Path
17990  Atelectasis|Effusion           1  ...   1024, 1024  D:\Minor Project\Dataset\data\images_004\image...
10051            No Finding           0  ...   1024, 1024  D:\Minor Project\Dataset\data\images_012\image...
1331             No Finding           0  ...   1024, 1024  D:\Minor Project\Dataset\data\images_003\image...
25371            No Finding           0  ...   1024, 1024  D:\Minor Project\Dataset\data\images_011\image...
10860              Effusion          32  ...   1024, 1024  D:\Minor Project\Dataset\data\images_008\image...

[5 rows x 12 columns]

//Checking the size after deletion
>>> i.getFileRows(mode=None, df=trimmed_df)
27896

//Trim your own dataframe like shown below
//before deletion
>>> df = i.openFile(mode='validation')
>>> i.getFileRows(mode=None, df=df)
27896

//new trimmed dataframe
>>> trimmed_df = i.deleteFileRows(mode=None, rows_to_delete=5, df=df)

//after deletion
>>> i.getFileRows(mode=None, df=trimmed_df)
27891

//Get Any no of rows/portion from the dataframe using the function below
//mode='master', open master csv file
//start_index=100, end_index=105, selects the rows with indexes starting from 100-104 
>>> trimmed_df = i.getChunkOfFile(start_index=100, end_index=105, mode='master')
>>> trimmed_df
                                            Image_Path            Finding_Labels  ...  Original_Pixel_Spacing_Y  Image_Shape
100  D:\Minor Project\Dataset\data\images_001\image...                No Finding  ...                     0.168   1024, 1024
101  D:\Minor Project\Dataset\data\images_001\image...     Effusion|Infiltration  ...                     0.168   1024, 1024
102  D:\Minor Project\Dataset\data\images_001\image...                No Finding  ...                     0.168   1024, 1024
103  D:\Minor Project\Dataset\data\images_001\image...  Infiltration|Mass|Nodule  ...                     0.168   1024, 1024
104  D:\Minor Project\Dataset\data\images_001\image...                  Fibrosis  ...                     0.168   1024, 1024

[5 rows x 12 columns]

//With your own dataframe
>>> trimmed_df = i.getChunkOfFile(start_index=100, end_index=105, mode=None, df=i.openFile(mode='test'))
>>> trimmed_df
    Finding_Labels  Follow_Ups  Patient_Id  ...  Original_Pixel_Spacing_Y Image_Shape                                         Image_Path
100     No Finding           0       27084  ...                     0.143  1024, 1024  D:\Minor Project\Dataset\data\images_011\image...
101     No Finding           2        8476  ...                     0.168  1024, 1024  D:\Minor Project\Dataset\data\images_004\image...
102     No Finding           0       26210  ...                     0.139  1024, 1024  D:\Minor Project\Dataset\data\images_011\image...
103   Infiltration           9        1250  ...                     0.168  1024, 1024  D:\Minor Project\Dataset\data\images_001\image...
104     No Finding           1        1733  ...                     0.168  1024, 1024  D:\Minor Project\Dataset\data\images_002\image...

[5 rows x 12 columns]


//Get Images from the dataframe using the function below
//my trimmed dataframe
>>> trimmed_df = i.getChunkOfFile(start_index=100, end_index=105, mode=None, df=i.openFile(mode='master'))

//get images in the trimmed dataframe
>>> images = i.getImages(df=trimmed_df)
>>> images.shape # here you can see the shape of returned images
(5, 1024, 1024, 1)

//Flatten the images as shown below
>>> images = i.getImages(df=trimmed_df, flatten=True)
>>> images.shape
(5, 1048576, 1)

//by default the function returns images as needed for neural networks, you can disable it as follows 
>>> images = i.getImages(df=trimmed_df, reshape_for_keras=False)
>>> images.shape
(5, 1024, 1024)

//flatten image arrays without changing the dimension
>>> images = i.getImages(df=trimmed_df, flatten=True, reshape_for_keras=False)
>>> images.shape
(5, 1048576)


//To see the images
//extract a chunk of master file
>>> trimmed_df = i.getChunkOfFile(start_index=100, end_index=105, mode=None, df=i.openFile(mode='master'))
//get images from the file
>>> images = i.getImages(df=trimmed_df)
//see the images 
>>> i.showImages(images)
//see images with interval between each image to 1s
>>> i.showImages(images, window_refresh_delay=1000)


//To normalize images
//'norm' variable must have a range in ascending order of exactly 2 numbers
//'dtype' variable gets applied at last, so it converts the dtype of array after normalizing to desired dtype  
//Normalize original image of uint8 0-255 to 0-1 float32 as follows
>>> i.showImages(i.getImages(df=trimmed_df), norm=(0, 1), dtype="float32")


//To see possible batch sizes for all the files
//This will work only when all the 4 files (master, train, test, validation) are present
>>> i.getBatchSizes()
{'train': [1, 66959], 'test': [1, 16741], 'validation': [1, 27901], 'master': [1, 7, 107, 149, 749, 1043, 15943, 111601]}


//If you want to find batch sizes for your own file or only one file using mode the see as shown below
>>> i.getBatchSizes(modes=['master']) # Mind here, its 'modes' with an 's' and not 'mode'
{'master': [1, 7, 107, 149, 749, 1043, 15943, 111601]}

or 

//Here since the function doesn't know the file type (we didnt use mode but used df) thats why it says 'Unknown File'
>>> i.getBatchSizes(modes=None, df=i.openFile(mode='master'))
{'Unknown File': [1, 7, 107, 149, 749, 1043, 15943, 111601]}



//If you went to trim the dataset to perfectly fit your batch_size (important for training neural networks), then do as follows
//batch size specified is 32 for master file
>>> new_fitted_master_df = i.shapeAccordingToBatchSize(batch_size=32, mode='master')

Total rows deleted to match the batch size = 17

>>> new_fitted_master_df
         index                                         Image_Path  ... Original_Pixel_Spacing_Y  Image_Shape
0            0  D:\Minor Project\Dataset\data\images_001\image...  ...                    0.143   1024, 1024
1            1  D:\Minor Project\Dataset\data\images_001\image...  ...                    0.143   1024, 1024
2            2  D:\Minor Project\Dataset\data\images_001\image...  ...                    0.168   1024, 1024
3            3  D:\Minor Project\Dataset\data\images_001\image...  ...                    0.171   1024, 1024
4            4  D:\Minor Project\Dataset\data\images_001\image...  ...                    0.143   1024, 1024
...        ...                                                ...  ...                      ...          ...
111579  111596  D:\Minor Project\Dataset\data\images_012\image...  ...                    0.168   1024, 1024
111580  111597  D:\Minor Project\Dataset\data\images_012\image...  ...                    0.168   1024, 1024
111581  111598  D:\Minor Project\Dataset\data\images_012\image...  ...                    0.168   1024, 1024
111582  111599  D:\Minor Project\Dataset\data\images_012\image...  ...                    0.168   1024, 1024
111583  111600  D:\Minor Project\Dataset\data\images_012\image...  ...                    0.168   1024, 1024

[111584 rows x 13 columns]

//Similarly as seen above in the examples, you can use your own dataframe too
>>> train_df = i.openFile(mode='train') #just train file used as a random csv file 
>>> new_fitted_train_df = i.shapeAccordingToBatchSize(batch_size=46, mode=None, df=train_df)

Total rows deleted to match the batch size = 29

>>> new_fitted_train_df
       index         Finding_Labels  Follow_Ups  ...  Original_Pixel_Spacing_Y  Image_Shape                                         Image_Path
0          0            Atelectasis           0  ...                  0.168000   1024, 1024  D:\Minor Project\Dataset\data\images_005\image...
1          1             No Finding           3  ...                  0.143000   1024, 1024  D:\Minor Project\Dataset\data\images_005\image...
2          2             No Finding           0  ...                  0.143000   1024, 1024  D:\Minor Project\Dataset\data\images_010\image...
3          3             No Finding           0  ...                  0.168000   1024, 1024  D:\Minor Project\Dataset\data\images_007\image...
4          4           Infiltration           1  ...                  0.194311   1024, 1024  D:\Minor Project\Dataset\data\images_003\image...
...      ...                    ...         ...  ...                       ...          ...                                                ...
66925  66954                 Nodule           1  ...                  0.194311   1024, 1024  D:\Minor Project\Dataset\data\images_012\image...
66926  66955             No Finding          13  ...                  0.143000   1024, 1024  D:\Minor Project\Dataset\data\images_010\image...
66927  66956                 Hernia           0  ...                  0.143000   1024, 1024  D:\Minor Project\Dataset\data\images_001\image...
66928  66957  Cardiomegaly|Effusion          24  ...                  0.168000   1024, 1024  D:\Minor Project\Dataset\data\images_003\image...
66929  66958               Fibrosis           5  ...                  0.143000   1024, 1024  D:\Minor Project\Dataset\data\images_006\image...

[66930 rows x 13 columns]

---------------------------------------------------------------------------------------------------------------------------------------------------
New Session:

//NOW COMES THE MOST IMPORTANT EXMPLE FOR WHICH THIS MODULE WAS CREATED
//Use the generator as shown below for training, testing, validating

from LoadImageDataset import ImageDatasetHandler as idh

batch_size = 32

i = idh("D:\\Minor Project\\Dataset\\data")

# Before using anything make sure you have created the all the 3 files (train, test, validation)

# Create generators as shown below
# if your neural network requires flattened images, set flatten=True when using the generators below
# You can also use your own processed dataframe (as shown in examples above) but is not recommended

train_gen = i.datasetGenerator(mode='train', batch_size=batch_size)

test_gen = i.datasetGenerator(mode='test', batch_size=batch_size)

validation_gen = i.datasetGenerator(mode='validation', batch_size=batch_size)

# Since the generator runs infinitely, we have to specify the no of steps at which keras will stop calling the generator for images
# find the no of steps shown below

train_steps_per_epoch = i.stepsPerEpoch(batch_size=batch_size, mode='train')

test_steps_per_epoch = i.stepsPerEpoch(batch_size=batch_size, mode='test')

val_steps_per_epoch = i.stepsPerEpoch(batch_size=batch_size, mode='validation')

# Use the keras fit_generator function as shown below, feel free to change the variables except the ones we declared above

# Assuming here that variable 'model' has the created neural network
my_model = model()

# Now, fit the model. Here, we use train and validation (optional) dataset
my_model.fit_generator(generator=train_gen, 
                       steps_per_epoch=train_steps_per_epoch,
                       epochs=16,
                       verbose=1,
                       validation_data=val_gen,
                       validation_steps=val_steps_per_epoch,
                       shuffle=True)

# Evaluate the model as shown below. Here, we use test dataset
test_accuracy = my_mnodel.evaluate_generator(generator=test_gen, steps=test_steps_per_epoch)

# Print the evaluated results
print("Test Accuracy =", test_accuracy[-1])


Hopefully, i was able to clear your doubts regarding how to use this library and how to use it with the neural networks.
If any doubts, contact me.

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Baaki ab suno ek poem,
Yeh voh button vali ki puri poem hai.


Usko Paana Hai To Poora Pagal Ban


Apni aankhon mein bhar kar le jaane hain
Mujhko uske aansoo kaam mein laane hai
Dekho hum koi vehshi nahin, deewane hain
Tumse batan khulwaane nahin lagwaane hai
Hum tum ek dooje ki seedhi hai jaana
Baaki duniya toh saapon ke khaane hain
Paqeeza cheezon ko paqeeza likho
Mat likho uski aankhen maikhaane hain

Teri nigaah-e-naaz se chhoote hue darkht
Mar jayen kya karen bata sookhe hue darkht
Hairat hain ped neem ke dene lage hain aam
Pagala gaye hain aapke choome hue darkht


Tere peechhe hogi duniya, pagal ban
Kya bola maine kuchh samajha?.. pagal ban
Sehara mein bhi dhoondh le dariya, pagal ban
Warna mar jayega pyaasa, pagal ban
Aadha daana aadha pagal, nahin nahin nahin
Usko paana hai to poora pagal ban
Daanai dikhlaane se kuch haasil nahi
Pagal khaana hai ye duniya, pagal ban
Dekhein tujhko log to pagal ho jayen
Itna umda itna aala pagal ban
Logon se dar lagta hai? to ghar mein baith
Jigra hai to mere jaisa pagal ban

Chaand, sitaare, phool, parinde, shaam, sawera ek taraf
Saari duniya uska charba uska chehra ek taraf
Wo lad kar bhi so jaye toh uska maatha choomu main
Usse muhabbat​ ek taraf hai usse jhagda ek taraf
Jis shay par wo ungli rakh de usko wo dilwaani hai
Uski khushiyaan sabse awval sasta mahnga ek taraf
Zakhmon par marham lagwao lekin uske haathon se
Chaara-saazi ek taraf hai uska chhoona ek taraf
Saari duniya jo bhi bole sab kuch shor-sharaaba hai
Sabka kahna ek taraf hai uska kahna ek taraf
Usne saari duniya maangi maine usko maanga hai
Uske sapne ek taraf hain mera sapna ek taraf


Jheelen kya hain?
Uski aankhen

Umda kya hai?
Uska chehara

Khushboo kya hai?
Uski saansein

Khushiya kya hain?
Uska hona

Toh gham kya hai?
Usse judaai

Saawan kya hai?
Uska rona

Sardi kya hai?
Uski udaasi

Garmi kya hai?
Uska gussa

Aur bahaaren?
Uska hansna

Meetha kya hai?
Uski baaten

Kadwa kya hai?
Meri baaten

Kya padhna hai?
Uska likkha

Kya sunna hai?
Uski ghazalen

Lab ki khwaahish?
Uska maatha

Zakhm ki khwaahish?
Uska chhoona

Dil ki khwaahish?
Usko paana

Duniya kya hai?
Ik jangal hai

Aur tum kya ho?
Ped samajh lo

Aur wo kya hai?
Ik raahi hai

Kya socha hai?
Usse muhabbat

Kya karte ho?
usse muhabbat

Iss ke alaawa?
Usse muhabbat

Matlab pesha?
Usse muhabbat

Usse muhabbat, Usse muhabbat, Usse muhabbat, Usse muhabbat ....

                                    – Varun Anand
