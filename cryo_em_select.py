from cgitb import small
from random import shuffle
from xml.etree.ElementPath import prepare_predicate
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
    LeakyReLU,
    GlobalAveragePooling2D,
    Flatten,
    concatenate, 
    Input
)
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from matplotlib import pyplot as plt


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
        #Y_batch = np.expand_dims(Y_batch, axis=3)
        
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
            #filename.touch(exist_ok=True)
            cv2.imwrite(str(filename), gauss_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return cropped_images, cropped_label_images
    
    def __len__(self):
        """
        Total number of batches.
        
        :return: The length of the 
        """
        return self.n // self.batch_size



class CryoEmNet:
    
    def __init__(self, batch_size, image_size, model=None):
        """
        :param batch_size: Batch size for training / prediction
        :param input_size: Input image size
        """
        self.batch_size = batch_size
        self.image_size = image_size
        if model is None:
            #self.model = self.build_preenc_convdec()
            #self.model = self.build_unet()
            self.model = self.small_unet()

            # Good base model
            #self.model = self.build_basic_model()

            # Prob good model (Need more training)
            # self.model = self.build_custom_unet()
            

        else:
            self.model = model

    def __convolution_layer(self, x, filters, kernel_size=3, padding='same', kernel_initializer='he_normal'):
        x = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        #x = Activation('relu')(x)
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
        
        output = Conv2D(1, 1, activation = 'sigmoid')(upsampling)
        
        model = Model(conv_base.input, output)


        num_base_layers = len(conv_base.layers)
        for layer in model.layers[:num_base_layers]:
            layer.trainable=False
        for layer in model.layers[num_base_layers:]:
            layer.trainable=True

        model.summary()

        return model
        

    def build_basic_model(self):
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
        convolution_5 = self.__convolution_layer(convolution_4,filters=32)
        upsampling_1 = UpSampling2D(size = (2,2))(convolution_5)
        
        convolution_6 = self.__convolution_layer(upsampling_1,filters=16)
        upsampling_2 = UpSampling2D(size = (2,2))(convolution_6)
        
        convolution_7 = self.__convolution_layer(upsampling_2,filters=8)
        upsampling_3 = UpSampling2D(size = (2,2))(convolution_7)

        outputs = Conv2D(1, 1, activation = 'sigmoid')(upsampling_3)

        # Specify model
        basic_model = Model(inputs=inputs, outputs=outputs)
        basic_model.summary()

        return basic_model

    
    def small_unet(self):
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
        upsampling_1 = UpSampling2D(size = (2,2))(convolution_4)
        merge_1 = concatenate([convolution_3,upsampling_1], axis = 3)
        upsampling_2 = UpSampling2D(size = (2,2))(merge_1)
        merge_2 = concatenate([convolution_2,upsampling_2], axis = 3)
        upsampling_3 = UpSampling2D(size = (2,2))(merge_2)
        merge_3 = concatenate([convolution_1,upsampling_3], axis = 3)

        outputs = Conv2D(1, 1, activation = 'sigmoid')(merge_3)

        # Specify model
        small_unet = Model(inputs=inputs, outputs=outputs)
        small_unet.summary()

        return small_unet


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
        convolution_5 = self.__convolution_layer(convolution_4,filters=32)
        upsampling_1 = UpSampling2D(size = (2,2))(convolution_5)
        merge_1 = concatenate([convolution_3,upsampling_1], axis = 3)
        convolution_6 = self.__convolution_layer(merge_1,filters=16)
        upsampling_2 = UpSampling2D(size = (2,2))(convolution_6)
        merge_2 = concatenate([convolution_2,upsampling_2], axis = 3)
        convolution_7 = self.__convolution_layer(merge_2,filters=8)
        upsampling_3 = UpSampling2D(size = (2,2))(convolution_7)
        merge_3 = concatenate([convolution_1,upsampling_3], axis = 3)

        outputs = Conv2D(1, 1, activation = 'sigmoid')(merge_3)

        # Specify model
        custom_unet = Model(inputs=inputs, outputs=outputs)
        custom_unet.summary()

        return custom_unet
    

    def build_convolutional(self):
        pass

    def build_unet(self):
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
        convolution_5 = self.__convolution_layer(convolution_4,filters=32)
        upsampling_1 = UpSampling2D(size = (2,2))(convolution_5)

        convolution_5_1 = self.__convolution_layer(upsampling_1,filters=32)
        convolution_5_2 = self.__convolution_layer(convolution_5_1,filters=32)

        merge_1 = concatenate([convolution_3,convolution_5_2], axis = 3)

        convolution_6 = self.__convolution_layer(merge_1,filters=16)
        upsampling_2 = UpSampling2D(size = (2,2))(convolution_6)

        convolution_6_1 = self.__convolution_layer(upsampling_2,filters=16)
        convolution_6_2 = self.__convolution_layer(convolution_6_1,filters=16)

        merge_2 = concatenate([convolution_2,convolution_6_2], axis = 3)

        convolution_7 = self.__convolution_layer(merge_2,filters=8)
        upsampling_3 = UpSampling2D(size = (2,2))(convolution_7)

        convolution_7_1 = self.__convolution_layer(upsampling_3,filters=16)
        convolution_7_2 = self.__convolution_layer(convolution_7_1,filters=16)

        merge_3 = concatenate([convolution_1,convolution_7_2], axis = 3)

        outputs = Conv2D(1, 1, activation = 'sigmoid')(merge_3)

        # Specify model
        custom_unet = Model(inputs=inputs, outputs=outputs)
        custom_unet.summary()

        return custom_unet

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
        import keras
        self.model.compile(
            # loss='mse'
            optimizer=opt, loss='mse', metrics=['accuracy']
        )
        self.model.fit(
            train_generator,
            epochs=epochs
        )

    
    def predict(self, img):
        # Predict should divide and conquer... and then assemble again
        
        result = self.model.predict(img)

        return result
        



    

    def show_predictions(self, image_name):
        import matplotlib.gridspec as gridspec

        path = str(os.getcwd()) + '/data_example/raw_data/' + str(image_name)
        image = cv2.imread(str(path))

        path = str(os.getcwd()) + '/data_example/label_data/' + str(image_name) + '-points.csv-gauss_img.jpg'
        label_image = cv2.imread(path)

        resized_images = []
        resized_label_images = []
        for i in range(224,1200,224):
            for j in range(224,1200,224):
                image_resize = image[i-224:i, j-224:j]
                image_resize = image_resize.astype(float)
                image_resize /= 255

                label_image_resize = label_image[i-224:i, j-224:j]
                label_image_resize= label_image_resize.astype(float)
                label_image_resize /= 255

                resized_images.append(image_resize)
                resized_label_images.append(label_image_resize)

        resized_images = np.asarray(resized_images)
        resized_label_images = np.asarray(resized_label_images)
        
        out = self.predict(resized_images[:,:,:,:])

        fig = plt.figure(figsize=(20, 8))
        
        outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.5)

        for i in range(len(resized_images)):
            inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                            subplot_spec=outer[i], wspace=0.2, hspace=0.5)

            ax = plt.Subplot(fig, inner[0])
            ax.imshow(resized_images[i,:,:,:], cmap='gray')
            ax.set_title('Input image', fontsize=10)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[1])
            ax.imshow(out[i,:,:,:],vmin=0,vmax=1, cmap='gray')
            ax.set_title('Predicted', fontsize=10)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)
            
            ax = plt.Subplot(fig, inner[2])
            ax.imshow(resized_label_images[i,:,:],vmin=0,vmax=1)
            ax.set_title('Ground truth', fontsize=10)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)
