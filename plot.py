import cryo_em_select as cryo
from keras.utils.vis_utils import plot_model
from pathlib import Path
import os
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = Path(os.getcwd()) / 'data_exploration'
    
    plt.figure(figsize=(20, 6))
    
    # summarize history for accuracy
    plt.subplot(141)
    img = cv2.imread(str(path / 'raw_data' / 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions.png'), cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title('Training Image')

    plt.subplot(142)
    img = cv2.imread(str(path / 'label_data' / 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions-points-points.jpg'))
    plt.imshow(img)
    plt.title('Label: Points')

    plt.subplot(143)
    img = cv2.imread(str(path / 'label_data' / 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions-points-gauss.jpg'), cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title('Label: Gaussian Circle')

    plt.subplot(144)
    img = cv2.imread(str(path / 'label_data' / 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions-points-white_square.jpg'), cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title('Label: White Square')

    plt.show()