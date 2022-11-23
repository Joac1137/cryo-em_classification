import cryo_em_select as cryo
from pathlib import Path
import os
import cv2

"""
Enable different labels
Delete labels/folder after each model are done
"""


def basic_model_experiment():
    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='points')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'basic_model_experiments' / 'basic_model' / 'points',
        learning_rate=10 ** -2,
        epochs=1,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='gauss')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'basic_model_experiments' / 'basic_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='white_square')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'basic_model_experiments' / 'basic_model' / 'white_square',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)


def custom_unet_experiment():
    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='points')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'small_unet_experiments' / 'small_unet_model' / 'points',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='gauss')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'small_unet_experiments' / 'small_unet_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='white_square')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'small_unet_experiments' / 'small_unet_model' / 'white_square',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)


def large_unet_experiment():
    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='points')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_unet_experiments' / 'large_unet_model' / 'points',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='gauss')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_unet_experiments' / 'large_unet_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='white_square')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_unet_experiments' / 'large_unet_model' / 'white_square',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)


def large_residual_unet_experiment():
    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='points')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_residual_unet_experiments' / 'large_residual_unet_model' / 'points',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='gauss')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_residual_unet_experiments' / 'large_residual_unet_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        640, 880, 1), label_type='white_square')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_residual_unet_experiments' / 'large_residual_unet_model' / 'white_square',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

def show_single(
    model, 
    filepath:Path=Path(str(os.getcwd())) / 'data' / 'raw_data' / 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions.png'
    ):
    """
    Function that shows the model prediction of a image

    :param model: The model we are using for prediction
    :param filepath: The path to the image we are using for prediction
    """
    from matplotlib import pyplot as plt
    import numpy as np

    original_image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)

    resized_image = original_image[21:-21, 40:-40]
    resized_image = np.expand_dims(resized_image, axis=0)

    prediction = model.predict(resized_image)

    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(original_image, cmap='gray')
    axarr[1].imshow(prediction[0], cmap='gray')

    plt.show()


def model_exploration():
    # Model explorations
    import keras.models
    label_type = 'gauss'
    path = Path(os.getcwd()) / 'server' / 'large' / 'Experiments' / 'large_residual_unet_experiments' / 'large_residual_unet_model' / label_type

    def dice_loss(y_true, y_pred, smooth=1):
        import keras.backend as K
        '''
        dice coefficient =2*sum(|y_true*y_pred|)/(sum(y_true^2)+sum(y_pred^2))
        
        Args:
        ->ground truth label
        ->predicted label
        -smooth:default is 1

        https://github.com/tensorflow/addons/pull/2558/commits/fa02a90d838b6e521c8f5b1ae2fd6c0a4bd2b794
        '''
        y_true = float(y_true)
        y_pred = float(y_pred)
        intersection = K.sum(K.abs(y_true*y_pred), axis=-1)
        return 1-(2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) - intersection + smooth)

    model = keras.models.load_model(str(path), custom_objects={'dice_loss':dice_loss})
    
    # cryo_thing = cryo.CryoEmNet(batch_size=20, image_size=(
    #     224, 224, 1), label_type=label_type, model=model)
    
    # image_path = Path('FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions.png')
    # cryo_thing.show_predictions(image_name=image_path, label_type=label_type)
    
    # show_single(model)
    
    with open(Path(str(os.getcwd())) / 'server' / 'large' / 'Experiments' /
        'basic_model_experiments' / 'basic_model' / 'white_square' / 'train_history', "rb") as f:
        import pickle
        history = pickle.load(f)
        show_history(history)

def show_history(history):
    """
    Method to show the history of the model

    :param history: History of the model form fitting
    """
    from matplotlib import pyplot as plt
    plt.figure(figsize=(20, 6))

    # summarize history for accuracy
    plt.subplot(131)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'val_accuracy'], loc='upper left')

    # # summarize history for loss
    plt.subplot(132)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')

    # summarize history for IoU
    plt.subplot(133)
    plt.plot(history['IoU'])
    plt.plot(history['val_IoU'])
    plt.title('model IoU')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.legend(['train_IoU', 'val_IoU'], loc='upper left')

    plt.show()

def main():
    # basic_model_experiment()
    # custom_unet_experiment()
    # large_unet_experiment()
    # large_residual_unet_experiment()
    
    model_exploration()
    
    
if __name__ == '__main__':
    main()
