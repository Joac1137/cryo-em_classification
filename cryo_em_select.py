from keras.utils import Sequence
from pathlib import Path
import os
import pandas as pd



class CryoBatchGenerator(Sequence):


    def __init__(self, batch_size, image_size=(224,224,3), shuffle=False, save_labels=False):
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.save_labels = save_labels

    def on_epoch_end(self):
        """
        Shuffle
        """
        if self.shuffle:
            pass

    def __getitem__(self, index):
        """
        Return image and labels

        Should save label data and not preprocess if we have create the labels
        """
        data_path = Path(str(os.getcwd()) + '/data_example/raw_data/')

        batch = [x for x in data_path.iterdir()][index * self.batch_size:(index + 1) * self.batch_size]
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

        print(label_path)
        print(label_df)
        



        pass 
    

    
    def __len__(self):
        pass







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


