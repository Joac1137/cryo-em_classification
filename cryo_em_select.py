from keras.utils import Sequence




class CryoBatchGenerator(Sequence):


    def __init__(self, X, batch_size, image_size=(224,224,3), shuffle=True):
        self.X = X
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        """
        Shuffle
        """
        if self.shuffle:
            pass

    def __getitem__(self):
        """
        Return image and labels

        Should save label data and not preprocess if we have create the labels
        """


        pass

    def __get_inputs(self, path):
        """
        This function takes the path for one image
        Remeber to sclace images properly
        """
    
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




