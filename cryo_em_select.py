from random import shuffle
from keras.utils import Sequence
from pathlib import Path
import os
import pandas as pd
import cv2
import preprocess
import numpy as np
from keras.layers import (
    Conv2D, 
    Activation, 
    BatchNormalization, 
    UpSampling2D, 
    Dense, 
    MaxPooling2D,
    concatenate, 
    Input
)
from keras.models import Model
from keras.optimizers import SGD, RMSprop


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

        # Collapse the first dimensions.
        X_batch = np.asarray(X_batch)
        X_batch = X_batch.reshape(-1, *X_batch.shape[-3:])
        Y_batch = np.asarray(Y_batch)
        Y_batch = Y_batch.reshape(-1, *Y_batch.shape[-2:])
        
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

        cropped_images = []
        cropped_label_images = []
        for i in range(224,1200,224):
            for j in range(224,1200,224):
                image = img[i-224:i, j-224:j]                
                image = image.astype(float)
                image /= 255.

                gauss_image = gauss_img[i-224:i, j-224:j]
                gauss_image = gauss_image.astype(float)
                gauss_image /= 255.

                cropped_images.append(image)
                cropped_label_images.append(gauss_image)


        if self.save_labels:
            filename = Path(str(os.getcwd()) + '/data_example/label_data/' + image_name + '-points.csv-gauss_img.jpg')
            filename.touch(exist_ok=True)
            #cv2.imwrite(str(filename), gauss_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return cropped_images, cropped_label_images
    
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
        #self.model = self.build_preenc_convdec()
        self.model = self.build_custom_unet()

    def __convolution_layer(self, x, filters, kernel_size=3, padding='same', kernel_initializer='he_normal'):
        x = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build_preenc_convdec(self):
        from keras.applications.mobilenet_v2 import MobileNetV2
        conv_base = MobileNetV2(weights='imagenet',
                      include_top=False,
                      input_shape=(224,224,3))

        # Decoder
        encoder = conv_base.output
        
        convolution_layer = self.__convolution_layer(encoder, filters=32)
        upsampling = UpSampling2D(size = (2,2))(convolution_layer)

        convolution_layer = self.__convolution_layer(upsampling, filters=16)
        upsampling = UpSampling2D(size = (2,2))(convolution_layer)

        convolution_layer = self.__convolution_layer(upsampling, filters=8)
        upsampling = UpSampling2D(size = (2,2))(convolution_layer)

        convolution_layer = self.__convolution_layer(upsampling, filters=4)
        upsampling = UpSampling2D(size = (2,2))(convolution_layer)

        convolution_layer = self.__convolution_layer(upsampling, filters=2)
        upsampling = UpSampling2D(size = (2,2))(convolution_layer)
        
        #feature_extractor = Model(conv_base.input, upsampling)

        output = Dense(64, activation="relu", name="denseL1")(upsampling)
        output = Dense(10, activation="relu", name="denseL2")(output)
        output = Dense(1, activation="sigmoid", name="denseL3")(output)
        
        model = Model(conv_base.input, output)


        num_base_layers = len(conv_base.layers)
        for layer in model.layers[:num_base_layers]:
            layer.trainable=False
        for layer in model.layers[num_base_layers:]:
            layer.trainable=True

        model.summary()

        return model
        

    def build_custom_unet(self):
        inputs = Input(shape=self.image_size)

        # Encoder
        convolution_1 = self.__convolution_layer(inputs,filters=8)
        pooling_1 = MaxPooling2D(pool_size=(2, 2))(convolution_1)
        convolution_2 = self.__convolution_layer(pooling_1,filters=16)
        pooling_2 = MaxPooling2D(pool_size=(2, 2))(convolution_2)
        convolution_3 = self.__convolution_layer(pooling_2,filters=32)
        pooling_3 = MaxPooling2D(pool_size=(2, 2))(convolution_3)
        convolution_4 = self.__convolution_layer(pooling_3,filters=64)

        # Decoder
        upsampling_7 = self.__convolution_layer(convolution_4,filters=32)
        upsampling_7 = UpSampling2D(size = (2,2))(upsampling_7)
        merge7 = concatenate([convolution_3,upsampling_7], axis = 3)
        upsampling_8 = self.__convolution_layer(merge7,filters=16)
        upsampling_8 = UpSampling2D(size = (2,2))(upsampling_7)
        merge8 = concatenate([convolution_2,upsampling_8], axis = 3)
        upsampling_9 = self.__convolution_layer(merge8,filters=8)
        upsampling_9 = UpSampling2D(size = (2,2))(upsampling_9)
        merge9 = concatenate([convolution_1,upsampling_9], axis = 3)

        # outputs = Dense(64, activation="relu", name="denseL1")(merge9)
        # outputs = Dense(10, activation="relu", name="denseL2")(outputs)
        # outputs = Dense(1, activation="sigmoid", name="denseL3")(outputs)

        # final_layer = Activation('sigmoid')(outputs)

        convolution_10 = Conv2D(1, 1, activation = 'sigmoid')(merge9)

        # Specify model
        convolution_model = Model(inputs=inputs, outputs=convolution_10)
        convolution_model.summary()

        return convolution_model
    

    def build_convolutional(self):
        pass

    def build_unet(self):
        pass

    def train(self, learning_rate=10 ** -2, epochs=10):
        
        data_path = [x for x in Path(str(os.getcwd()) + '/data_example/raw_data/').iterdir()]

        train_generator = CryoBatchGenerator(
            X=data_path,
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=True,
            save_labels=True
        )

        optimizer = SGD(
            learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True
        )

        opt = RMSprop(
            learning_rate=learning_rate
        )

        self.model.compile(
            optimizer=opt, loss='mse', metrics=['accuracy']
        )
        self.model.fit(
            train_generator,
            epochs=epochs
        )

    
    def predict(self, img):
        result = self.model.predict(img)

        return result
        # Predict should divide and conquer... and then assemble again
        