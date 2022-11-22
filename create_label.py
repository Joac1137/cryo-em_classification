import os
from pathlib import Path
import pandas as pd
import cv2
import preprocess

if __name__ == '__main__':
    
    label_path = Path(
        str(os.getcwd())) / 'data_exploration' / 'label_annotation' / 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions-points.csv'
    if label_path.exists():
        
        label_df = pd.read_csv(label_path, header=None)
        label_df = label_df.round(0).astype(int)
    
    # Coordinate system turns in a weird way.
    points = [(rows[1], rows[0]) for _, rows in label_df.iterrows()]

    # Load greyscale to remove the 3 channels
    path = Path(os.getcwd()) / 'data_exploration' / 'raw_data' / 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions.png'
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    gauss_img = preprocess.GaussianHighlight(
        img[:, :], points, 32, label_type='white_square')


    filename = Path(str(os.getcwd()) +
                    '/data_exploration/label_data/' + 'FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions-points' + '-white_square.jpg')
    if not filename.exists():
        cv2.imwrite(str(filename), gauss_img, [
                    cv2.IMWRITE_JPEG_QUALITY, 100])
