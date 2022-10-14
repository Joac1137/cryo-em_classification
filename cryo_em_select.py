from fileinput import filename
from keras.utils import Sequence
from pathlib import Path
import os
import pandas as pd
import cv2
import preprocess
import numpy as np


class CryoBatchGenerator(Sequence):


    def __init__(self, X, batch_size, image_size=(224,224,3), shuffle=False, save_labels=False):
        self.X = X
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.save_labels = save_labels
        self.n = len(X)

    def on_epoch_end(self):
        """
        Shuffle
        """
        if self.shuffle:
            np.random.shuffle(self.X)

    def __getitem__(self, index):
        """
        Return image and labels

        Should save label data and not preprocess if we have create the labels
        """

        batch = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__get_data(batch)
        return X, Y


    def __get_data(self, batch):
        """
        Get resized train and label images
        """

        X_batch, Y_batch = [self.__get_input(x) for x in batch]
        
        return X_batch, Y_batch


        
    def __get_input(self, path):
        """
        This function takes the path for one image
        Remeber to sclace images properly
        """
        image_name = os.path.basename(os.path.normpath(path))
        label_path = Path(str(os.getcwd()) + '/data_example/label_annotation/' + image_name + '-points.csv')

        label_df = pd.read_csv(label_path, header=None)
        label_df = label_df.round(0).astype(int)

        # Coordinate system turns in a weird way.
        points = [(rows[1], rows[0]) for index, rows in label_df.iterrows()]

       
        img = cv2.imread(os.path.join("",path))
        gauss_img = preprocess.GaussianHighlight(img[:,:,0], points, 60)
    
        if self.save_labels:
            filename = Path(str(os.getcwd()) + '/data_example/label_data/' + image_name + '-points.csv')
            print("Saving labels for file {file_name}").format(filename=image_name)

            cv2.imwrite(filename, gauss_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return img/255., gauss_img/255.

    
    def __len__(self):
        return self.n // self.batch_size







class CryoEmNet:
    
    def __init__(self):
        pass

    
    def build_convolutional(self):
        pass


    def build_unet(self):
        pass

    def train(self):
        pass
    
    def predict(self):
        # Predict should divide and conquer... and then assemble again

        pass


