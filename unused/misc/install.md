For this install you must have Anaconda (https://www.anaconda.com/distribution/) or Miniconda(https://docs.conda.io/en/latest/miniconda.html) downloaded and installed on your machine

After conda is installed run these commands through command prompt on your system (you can name your environment whatever you want, for this guide it has been named 'deployment')

You can use updated versions of these modules if you choose. The version specified here are what was used to create this software

$ conda create --name deployment --y

$ conda activate deployment

$ conda install python=3.7 --y

$ pip install opencv-python==4.2.0.32

$ pip install pyqt5==5.14.1

$ pip install scipy==1.4.1

$ pip install filterpy==1.4.5

$ pip install tqdm==4.42.1

$ conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.1 -c pytorch --y

Then you can run 'python App.py' using this conda environment to execute our software while in the source_code directory