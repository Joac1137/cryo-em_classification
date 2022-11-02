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
    LeakyReLU,
    GlobalAveragePooling2D,
    Flatten,
    concatenate, 
    Input
)
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard,
    LearningRateScheduler
)


class CryoBatchGenerator(Sequence):
    """
    This is a generator for batches of training data. It reads in a list of paths to the training images. We preprocess the images 
    and create the labels from the .csv files. 
    """

    def __init__(self, X, batch_size, image_size=(224,224,1), shuffle=False, save_labels=False, gauss_label=False):
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
        self.gauss_label = gauss_label
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
        X_batch = np.concatenate(X_batch, axis=0)

        Y_batch = np.asarray(Y_batch)
        Y_batch = np.concatenate(Y_batch,axis=0)


        return X_batch, Y_batch

    def __get_input(self, path):
        """
        This method labels a image given by its path. 

        :param path: The path to the image file
        :return: Return images
                    First output is the original image scaled
                    Second output is the labels for the images
        """
        image_height = 622
        image_width = 900

        label_path = Path(str(os.getcwd()) + '/data/label_annotation/' + path.stem + '-points' + '.csv')
        if not label_path.exists():
            return [],[]
        label_df = pd.read_csv(label_path, header=None)
        label_df = label_df.round(0).astype(int)

        # Coordinate system turns in a weird way.
        points = [(rows[1], rows[0]) for _, rows in label_df.iterrows()]
    
        #Load greyscale to remove the 3 channels
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        gauss_img = preprocess.GaussianHighlight(img[:,:], points, 32, self.gauss_label)

        img = img[30:-30,30:-30]
        gauss_img = gauss_img[30:-30,30:-30]
        # Apply edge detection
        #img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)

        cropped_images = []
        cropped_label_images = []
        # 682, 960
        for i in range(self.image_size[0],image_height,self.image_size[0]):
            for j in range(self.image_size[1],image_width,self.image_size[1]):
                image = img[i-self.image_size[0]:i, j-self.image_size[1]:j]                
                image = image.astype(float)
                image /= 255.
                # Zero center image
                image = (image - image.mean()) / image.std()
                #image = (2 * image) - 1

                gauss_image = gauss_img[i-self.image_size[0]:i, j-self.image_size[1]:j]
                gauss_image = gauss_image.astype(float)
                gauss_image /= 255.

                cropped_images.append(image)
                cropped_label_images.append(gauss_image)


        if self.save_labels:
            filename = Path(str(os.getcwd()) + '/data/label_data/' + path.stem + '-points.jpg')
            if not filename.exists():
                cv2.imwrite(str(filename), gauss_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return cropped_images, cropped_label_images
    
    def __len__(self):
        """
        Total number of batches.
        
        :return: The length of the 
        """
        return self.n // self.batch_size



class CryoEmNet:
    """
    This is our cryo-em segmentation class
    """
    
    def __init__(self, batch_size, image_size=(224,224,1), model=None, gauss_label=False):
        """
        :param batch_size: Batch size for training / prediction
        :param input_size: Input image size
        """
        self.batch_size = batch_size
        self.image_size = image_size
        self.gauss_label = gauss_label

        if model is None:
            # self.model = self.build_preenc_convdec()
            # self.model = self.build_unet()
            # self.model = self.small_unet()

            # Good base model
            # self.model = self.build_basic_model()

            # Prob good model (Need more training)
            # self.model = self.build_custom_unet()
            
            self.model = self.build_large_unet()
            

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

    
    def build_large_unet(self):
        inputs = Input(shape=self.image_size)

        # Eacly have larger filters and then reduce the size in later layers
        # Maybe apply 1x1 convolution in order to do dimension reduction
        
        # Maybe apply residual connections

        # Apply skip connection in order to fix vanishing gradiant problem

        # Encoder
        # Layer 1
        convolution_1 = self.__convolution_layer(inputs,filters=8)
        pooling_1 = MaxPooling2D(pool_size=(2, 2))(convolution_1)

        # Layer 2
        convolution_2 = self.__convolution_layer(pooling_1,filters=16)
        pooling_2 = MaxPooling2D(pool_size=(2, 2))(convolution_2)

        # Layer 3
        convolution_3 = self.__convolution_layer(pooling_2,filters=32)

        # Layer 4
        convolution_4 = self.__convolution_layer(convolution_3,filters=16)

        # Layer 5
        convolution_5 = self.__convolution_layer(convolution_4,filters=32)
        pooling_3 = MaxPooling2D(pool_size=(2,2))(convolution_5)

        # Layer 6
        convolution_6 = self.__convolution_layer(pooling_3,filters=64)

        # Layer 7
        convolution_7 = self.__convolution_layer(convolution_6,filters=32)

        # Layer 8
        convolution_8 = self.__convolution_layer(convolution_7, filters=64)
        pooling_4 = MaxPooling2D(pool_size=(2,2))(convolution_8)

        # Layer 9
        convolution_9 = self.__convolution_layer(pooling_4, filters=128)

        # Layer 10
        convolution_10 = self.__convolution_layer(convolution_9, filters=64)
        
        # Layer 11
        convolution_11 = self.__convolution_layer(convolution_10, filters=128)

        # Layer 12
        convolution_12 = self.__convolution_layer(convolution_11, filters=64)
        
        # Layer 13
        convolution_13 = self.__convolution_layer(convolution_12, filters=128)

        # Layer 14
        convolution_14 = self.__convolution_layer(convolution_13, filters=64)

        # Layer 15
        convolution_15 = self.__convolution_layer(convolution_14, filters=128)

        # Decoder
        # Layer 16
        convolution_16 = self.__convolution_layer(convolution_15, filters=64)
        upsampling_1 = UpSampling2D(size = (2,2))(convolution_16)
        merge_1 = concatenate([convolution_7,upsampling_1], axis = 3) 

        # Layer 17
        convolution_17 = self.__convolution_layer(merge_1, filters=32)

        # Layer 18
        convolution_18 = self.__convolution_layer(convolution_17, filters=64)

        # Layer 19
        convolution_19 = self.__convolution_layer(convolution_18, filters=32)
        upsampling_2 = UpSampling2D(size = (2,2))(convolution_19)
        merge_2 = concatenate([convolution_4,upsampling_2], axis = 3) 

        # Layer 20
        convolution_20 = self.__convolution_layer(merge_2, filters=16)
        upsampling_3 = UpSampling2D(size = (2,2))(convolution_20)
        merge_3 = concatenate([convolution_2,upsampling_3], axis = 3) 

        # Layer 21
        convolution_21 = self.__convolution_layer(merge_3, filters=16)
        upsampling_4 = UpSampling2D(size = (2,2))(convolution_21)
        merge_4 = concatenate([convolution_1,upsampling_4], axis = 3)


        outputs = Conv2D(1, 1, activation = 'sigmoid')(merge_4)
        
        # Specify model
        large_unet = Model(inputs=inputs, outputs=outputs)
        large_unet.summary()

        return large_unet
    
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

    def train(
        self, 
        filepath: Path=Path(str(os.getcwd())) / 'model' / 'checkpoint',
        nb_epoch_early=10,
        warmrestarts=True,
        learning_rate=10 ** -2, 
        epochs=10,
        save_log=True,
        save_model=True
    ):
        
        data_path = [x for x in Path(str(os.getcwd()) + '/data/raw_data/').iterdir()]

        train_generator = CryoBatchGenerator(
            X=data_path[:100],
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=True,
            save_labels=True,
            gauss_label=self.gauss_label
        )

        # Define callbacks
        all_callbacks = []
        if save_model:
            checkpoint = ModelCheckpoint(
                filepath=filepath,
                monitor="accuracy",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                save_freq='epoch',
            )
            all_callbacks.append(checkpoint)

        early_stop = EarlyStopping(
            monitor="accuracy",
            min_delta=0.0005,
            patience=nb_epoch_early,
            mode="max",
            verbose=1,
        )
        all_callbacks.append(early_stop)

        reduceLROnPlateau = ReduceLROnPlateau(
            monitor="accuracy",
            factor=0.1,
            patience=int(nb_epoch_early * 0.6),
            verbose=1,
        )
        all_callbacks.append(reduceLROnPlateau)
        if save_log:
            try:
                os.makedirs(os.path.expanduser("logs/"))
            except:
                pass

            tb_counter = (
                len(
                    [
                        log
                        for log in os.listdir(os.path.expanduser("logs/"))
                        if "cinderella" in log
                    ]
                )
                + 1
            )
            tensorboard = TensorBoard(
                log_dir=os.path.expanduser("logs/") + "cinderella" + "_" + str(tb_counter),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
            )
            all_callbacks.append(tensorboard)
        if not warmrestarts:
            reduce_lr_on_plateau = ReduceLROnPlateau(
                monitor="accuracy",
                factor=0.1,
                patience=int(nb_epoch_early * 0.6),
                verbose=1,
            )
            all_callbacks.append(reduce_lr_on_plateau)
        else:
            def scheduler(epochs, learning_rate):
                if epochs < 10:
                    return learning_rate
                else:
                    import tensorflow as tf
                    return learning_rate * tf.math.exp(-0.1)
            schedule = LearningRateScheduler(
                schedule=scheduler,
                verbose=True
            )
            
            all_callbacks.append(schedule)

        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        self.model.compile(
            optimizer=optimizer, 
            # Loss='binary_crossentropy'
            loss='mse', 
            metrics=['accuracy'],
        )

        self.model.fit(
            train_generator,
            epochs=epochs,
            callbacks=all_callbacks
        )


        folder = Path(os.getcwd()) / 'data' / 'label_data'
        for filename in folder.listdir():
            file_path = folder / Path(filename)
            try:
                if file_path.exists():
                    os.remove(str(file_path))
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def predict(self, img):
        # Predict should divide and conquer... and then assemble again
        
        result = self.model.predict(img)

        return result
        



    

    def show_predictions(self, image_name:Path):
        import matplotlib.gridspec as gridspec
        image_height = 622
        image_width = 900

        path = str(os.getcwd()) + '/data/raw_data/' + image_name.name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        path = str(os.getcwd()) + '/data/label_data/' + image_name.stem + '-points' + '.jpg'
        label_image = cv2.imread(str(path))

        resized_images = []
        resized_label_images = []
        # 682, 960
        for i in range(self.image_size[0],image_height,self.image_size[0]):
            for j in range(self.image_size[1],image_width,self.image_size[1]):
                image_resize = image[i-self.image_size[0]:i, j-self.image_size[1]:j]
                image_resize = image_resize.astype(float)
                image_resize /= 255.
                # Zero center image
                image_resize = (image_resize - image_resize.mean()) / image_resize.std()

                label_image_resize = label_image[i-self.image_size[0]:i, j-self.image_size[1]:j]
                label_image_resize= label_image_resize.astype(float)
                label_image_resize /= 255.

                resized_images.append(image_resize)
                resized_label_images.append(label_image_resize)

        resized_images = np.asarray(resized_images)    
        resized_label_images = np.asarray(resized_label_images)
        
        out = self.predict(resized_images[:,:,:])

        fig = plt.figure(figsize=(20, 8))
        
        outer = gridspec.GridSpec(image_width // self.image_size[1], image_height // self.image_size[0], wspace=0.2, hspace=0.5)

        for i in range(len(resized_images)):
            inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                            subplot_spec=outer[i], wspace=0.2, hspace=0.5)

            ax = plt.Subplot(fig, inner[0])
            ax.imshow(resized_images[i,:,:], cmap='gray')
            ax.set_title('Input image', fontsize=10)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[1])
            ax.imshow(out[i,:,:],vmin=0,vmax=1, cmap='gray')
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
