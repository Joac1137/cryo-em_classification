import os
import starfile
from os import path

# Set the correct values for this path
images_folder = "/emcc/emfac/jonathan-picker/emfac_emfac_cs_0_20201016_FER17-4_data/relion31/MotionCorr/job002/emcc/anbog/ferritin/emfac_emfac_cs_0_20201016_FER17-4_data/tiff_all"
autopick_folder = "/emcc/emfac/jonathan-picker/emfac_emfac_cs_0_20201016_FER17-4_data/relion31/AutoPick/job024/emcc/anbog/ferritin/emfac_emfac_cs_0_20201016_FER17-4_data/tiff_all"
output_folder = "/u/au605475/Desktop/output"
downsize_scale = 6

# scan all files in images folder
print("Scanning dir")
files = os.listdir(images_folder)

processing_file_num = 1

for relativeFilePath in files:
    processing_file_num_string = f'{processing_file_num}'.zfill(4)

    # Find mrc files in images folder
    filePath = images_folder + "/" + relativeFilePath

    filePathParts = filePath.split("/")

    fileFullName = filePathParts[len(filePathParts) - 1]
    [fileName, fileExt] = fileFullName.split(".")

    # paths
    star_file_path = autopick_folder + "/" + fileName + "_autopick.star"

    # Check that is the correct mrc file
    if fileExt == "mrc" and fileName[-2:] != "PS" and path.exists(star_file_path):
        print(f'{processing_file_num_string}: Processing {fileName}')
        processing_file_num = processing_file_num + 1

        # process image
        print(f'   Processing image')
        imageOutPath = output_folder + "/" + fileName + ".png"
        os.system(
            f'mrc2any -f PNG -b {downsize_scale} {filePath} {imageOutPath}')

        # process data points

        print(f'   Processing data points')
        dataOutPath = output_folder + "/" + fileName + "-points.csv"
        particles = starfile.read(star_file_path)

        pointsLength = len(particles['rlnCoordinateX'])

        print(f'      Found {pointsLength} points')

        points = []

        # Loop through all the points and insert x and y into points array
        for index in range(len(particles['rlnCoordinateX'])):
            starPathParts = star_file_path.split("/")

            starFullName = starPathParts[len(starPathParts) - 1]
            [starName, imageExt] = starFullName.split(".")

            point = {
                'x':  round(particles['rlnCoordinateX'][index] / downsize_scale),
                'y':  round(particles['rlnCoordinateY'][index] / downsize_scale),
            }

            points.append(point)

        # Write out all the data points to file
        stringData = '\n'.join(list(map(lambda point: str(
            point['x']) + "," + str(point['y']), points)))
        f = open(dataOutPath, "w")
        f.write(stringData)
        f.close()
