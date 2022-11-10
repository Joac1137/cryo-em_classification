import cryo_em_select as cryo
from pathlib import Path
import os

"""
Enable different labels
Delete labels/folder after each model are done
"""


def basic_model_experiment():
    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='points')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'basic_model_experiments' / 'basic_model' / 'points',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='gauss')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'basic_model_experiments' / 'basic_model' / 'gauss',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='white_square')
    model.build_basic_model()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'basic_model_experiments' / 'basic_model' / 'white_square',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)

def custom_unet_experiment():
    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='points')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'small_unet_experiments' / 'small_unet_model' / 'points',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='gauss')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'small_unet_experiments' / 'small_unet_model' / 'gauss',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='white_square')
    model.build_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'small_unet_experiments' / 'small_unet_model' / 'white_square',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)

def large_unet_experiment():
    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='points')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'large_unet_experiments' / 'large_unet_model' / 'points',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='gauss')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'large_unet_experiments' / 'large_unet_model' / 'gauss',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='white_square')
    model.build_large_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'large_unet_experiments' / 'large_unet_model' / 'white_square',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)

def large_residual_unet_experiment():
    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='points')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'large_residual_unet_experiments' / 'large_residual_unet_model',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='gauss')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'large_residual_unet_experiments' / 'large_residual_unet_model',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1), label_type='white_square')
    model.build_large_residual_unet()

    model.train(
        filepath=Path(str(os.getcwd())) / 'Experiments' / 'large_residual_unet_experiments' / 'large_residual_unet_model',
        learning_rate=10 ** -2, 
        epochs=100,
        save_log=True,
        save_model=True)


def main():
    basic_model_experiment()
    custom_unet_experiment()
    large_unet_experiment()
    large_residual_unet_experiment()

if __name__ == '__main__':
    main()