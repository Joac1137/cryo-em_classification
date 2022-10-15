from tkinter import image_names
from keras.utils import Sequence
from pathlib import Path
import os
import pandas as pd
import cv2
import preprocess
import numpy as np
from keras.layers import Conv2D, Activation, BatchNormalization



class CryoBatchGenerator(Sequence):
    """
    This is a generator for batches of training data. It reads in a list of paths to the training images. We preprocess the images 
    and create the labels from the .csv files. 
    """

    def __init__(self, X, batch_size, image_size=(224,224,3), shuffle=False, save_labels=False):
        """
        Initialize a batch generator.

        :param X: List of image paths
        :param batch_size: Number of images in each batch
        :param image_size: Dimension of each image
        :param shuffle: Whether to shuffle the images
        :param save_labels: Whether to save the labels for the images
        """
        
        self.X = X
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.save_labels = save_labels
        self.n = len(X)

    def on_epoch_end(self):
        """
        This method shuffle the training data at the end of a epoch.

        :return: None
        """
        if self.shuffle:
            np.random.shuffle(self.X)

    def __getitem__(self, index):
        """
        This method generate one batch of data.

        :param index: index of training data
        :return: Return a batch.
                    First output is the training data
                    Second output is the label for the images
        """

        batch = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__get_data(batch)

        return X, Y

    def __get_data(self, batch):
        """
        This method extracts the labels for all images in the given batch.

        :param batch: Batch of image paths
        :return: Return a batch.
                    First output is the raw images
                    Second output is the label for the images
        """

        # __get_input return tuples, thus we convert them to list of tuples and further map to a list
        X_batch, Y_batch = zip(*[self.__get_input(x) for x in batch])
        X_batch = np.asarray(X_batch)
        Y_batch = np.asarray(Y_batch)
        
        return X_batch, Y_batch

    def __get_input(self, path):
        """
        This method labels a image given by its path. 

        :param path: The path to the image file
        :return: Return images
                    First output is the original image scaled
                    Second output is the labels for the images
        """
        image_name = os.path.basename(os.path.normpath(path))
        label_path = Path(str(os.getcwd()) + '/data_example/label_annotation/' + image_name + '-points.csv')

        label_df = pd.read_csv(label_path, header=None)
        label_df = label_df.round(0).astype(int)

        # Coordinate system turns in a weird way.
        points = [(rows[1], rows[0]) for _, rows in label_df.iterrows()]
       
        img = cv2.imread(os.path.join("",path))
        gauss_img = preprocess.GaussianHighlight(img[:,:,0], points, 60)
    
        if self.save_labels:
            filename = Path(str(os.getcwd()) + '/data_example/label_data/' + image_name + '-points.csv-gauss_img.jpg')
            print("Saving labels for file {image_name}".format(image_name=image_name))

            cv2.imwrite(str(filename), gauss_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return img/255., gauss_img/255.
    
    def __len__(self):
        """
        Total number of batches.
        
        :return: The length of the 
        """
        return self.n // self.batch_size



class CryoEmNet:
    
    def __init__(self, batch_size, image_size):
        """
        :param batch_size: Batch size for training / prediction
        :param input_size: Input image size
        """
        self.batch_size = batch_size
        self.image_size = image_size
        
    def __convolution_layer(x, filters, kernel_size=3, padding='same', kernel_initializer='he_normal'):
        x = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build_convolutional(self):
        pass


    def build_unet(self):
        pass

    def train(self):
        pass
    
    def predict(self):
        # Predict should divide and conquer... and then assemble again

        pass


