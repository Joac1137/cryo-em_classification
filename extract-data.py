import os
from pathlib import Path

p = Path(os.getcwd()) / 'data' / 'output'

for file in p.iterdir():
    if file.suffix == '.csv':
        os.rename(str(file), os.path.join(
            os.getcwd(), 'data/label_annotation', str(file.name)))
    else:
        os.rename(str(file), os.path.join(
            os.getcwd(), 'data/raw_data', str(file.name)))
