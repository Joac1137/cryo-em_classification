# Cryo-em Segmentation
Cryo-electron microscopy (Cryo-EM) is a way of imaging very small particle, but particle picking from these images remains a challenging early step in the Cryo-EM pipeline due to the diversity of particle shapes and the extremely low signal-to-noise ratio. Because of these issues, significant human intervention is often required to generate a high-quality set of particles. We want to apply the power of deep neural networks in order to alleviate this human intervention.


# Getting Started
- Create virtual environment - `python -m venv name_of_virtual_environment`
- Activate virtual environment - `name_of_virtual_environment\Scripts\activate`
- Choose the virtual environment as interpreter in VS Code - `CTRL + SHIFT + p` and select python.exe from the virtual environment
- Install requirements - `pip install -r requirements.txt`

_Note_: Some package might not install because of _Windows Long Paths_. Solved [here](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell).

- Select virtual environment for ipynb kernel

- Freeze requirements - `pip freeze > requirements.txt`


# How to run

To train the network simply run `python runner.py`.

To explore the result run `python runner.py -e`.
You are able to add the following arguments:
* `-p <path>` The path to the model (default: ./server/small/Experiments/large_residual_unet_experiments/large_residual_unet_model)
* `-l <label_type>` The label type to explore (default: white_square): white_square, guass, points 
* `-i <image_file>` The path of the image to check (default: FoilHole_16384305_Data_16383479_16383481_20201016_164256_fractions.png)


# Small Assignemnet - Brief project status
- Have you acquired and prepared your dataset? <br>
    We currently have 3256 cryo em data images. Each individual image is currently 682x960 and we therefore further divide each image into 224x224, such that the model doesn't have to large of an input. We are further working with the data because currently we have no labels on the edge of each image. Thereby, we might crop each of the 3256 images in order to increase the performance. <br>
    We are also considering using different types of labels for the images. Initially we just want to completely color the points, but we might experiment using a gaussian distribution on the points instead.

- Have you set up an appropriate neural network to solve your task? <br>
    Currently we are working with a number of different models in order to compare their results. First, we have build a base segmentation model, which doesn't have Unet connections. Furthermore, we have a small Unet model and a larger one. 
    We are also considering using a pretrained encoder in order to compare the performance. This was also our motivation to use images of size 224x224 because this was demanded from the pretrained model. 

- Have you trained and tested your first network? <br>
    We have tried to train the different models, but haven't setup the experiments in a systemized manner yet. The performance increases when training with the base_model, small_unet and large_unet respectively. We have ensured accuracy of up to 92% (where we currently are only training on 100/3256 images). <br>
    The performance is drastically reduced when using the gaussian distribution on the points instead. We are working on how we might fix this. 

- What are the next steps? <br>
    The next steps is definitely to setup the experiments in a more systematic manner. We then further aim to train all the different models on the entires dataset on a server that we have available. Thereafter, we might just analyze, finetune and rerun the experiments and then begin the report. 
    We have also considered using more residual connections and further add 1x1 convolutions. 

- Any uncertainties or issues you need help with? <br>
    When designing the large_unet it is very difficult to determine the specific number of layers that we should include. Is there a rule of thumb or do we just have to try and observe the results?

If you are interested in observing the results, the code is available at [Github](https://github.com/Joac1137/cryo-em_segmentation). We have different logs that can be viewed using tensorboard <br>
<code> 
    %load_ext tensorboard <br>
    %tensorboard --logdir logs/cinderella_31/train 
</code> <br>
Image of differnet preliminary experiments are located in the `Experiments` folder. 