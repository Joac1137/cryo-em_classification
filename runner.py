import cryo_em_select as cryo
from pathlib import Path
import os


def main():
    model = cryo.CryoEmNet(batch_size=4, image_size=(224,224,1))
    model.train(
        filepath=Path(str(os.getcwd())) / 'test_model' / 'checkpoint',
        learning_rate=10 ** -2, 
        epochs=20,
        save_log=True,
        save_model=True)

if __name__ == '__main__':
    main()