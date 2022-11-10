import cryo_em_select as cryo
from pathlib import Path
import os
from pathlib import Path

model = cryo.CryoEmNet(batch_size=4, image_size=(
    224, 224, 1), gauss_label=True)
model.train(
    filepath=Path(str(os.getcwd())) / 'test_model' / 'checkpoint',
    learning_rate=10 ** -2,
    epochs=20,
    save_log=False,
    save_model=False)


path = Path(
    'FoilHole_16384338_Data_16383479_16383481_20201016_164621_fractions.png')
model.show_predictions(path)
