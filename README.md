# cryo-em_segmentation
Cryo-electron microscopy (Cryo-EM) is a way of imaging very small particle, but particle picking from these images remains a challenging early step in the Cryo-EM pipeline due to the diversity of particle shapes and the extremely low signal-to-noise ratio. Because of these issues, significant human intervention is often required to generate a high-quality set of particles. We want to apply the power of deep neural networks in order to alleviate this human intervention.


# Getting Started
- Create virtual environment - `python -m venv name_of_virtual_environment`
- Activate virtual environment - `name_of_virtual_environment\Scripts\activate`
- Choose the virtual environment as interpreter in VS Code - `CTRL + SHIFT + p` and select python.exe from the virtual environment
- Install requirements - `pip install -r requirements.txt`

_Note_: Some package might not install because of _Windows Long Paths_. Solved [here](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell).


- Freeze requirements - `pip freeze > requirements.txt`