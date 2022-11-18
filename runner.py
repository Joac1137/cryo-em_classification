import cryo_em_select as cryo
from pathlib import Path
import os

"""
Enable different labels
Delete labels/folder after each model are done
"""


def basic_model_experiment():
    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='points')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'basic_model_experiments' / 'basic_model' / 'points',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='gauss')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'basic_model_experiments' / 'basic_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='white_square')
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
        224, 224, 1), label_type='points')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'small_unet_experiments' / 'small_unet_model' / 'points',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='gauss')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'small_unet_experiments' / 'small_unet_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='white_square')
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
        224, 224, 1), label_type='points')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_unet_experiments' / 'large_unet_model' / 'points',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='gauss')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_unet_experiments' / 'large_unet_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='white_square')
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
        224, 224, 1), label_type='points')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_residual_unet_experiments' / 'large_residual_unet_model' / 'points',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='gauss')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' /
        'large_residual_unet_experiments' / 'large_residual_unet_model' / 'gauss',
        learning_rate=10 ** -2,
        epochs=30,
        save_log=True,
        save_model=True)

    model = cryo.CryoEmNet(batch_size=20, image_size=(
        224, 224, 1), label_type='white_square')
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
    import matplotlib.image as mpimg
    import numpy as np

    original_image = mpimg.imread(str(filepath))

    resized_image = original_image[21:-21, 40:-40]
    resized_image = np.expand_dims(resized_image, axis=0)

    prediction = model.predict(resized_image)

    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(original_image)
    axarr[1].imshow(prediction[0], cmap='gray')

    plt.show()
    
def main():
    basic_model_experiment()
    custom_unet_experiment()
    large_unet_experiment()
    large_residual_unet_experiment()
    
    # Model explorations
    # import keras.models
    # model = keras.models.load_model('path/to/location')
    # show_single(model)

if __name__ == '__main__':
    main()
