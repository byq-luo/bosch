The following commands install cudatoolkit-10.1.243.
You need to install the cuda development files to '/usr/local/cuda-10.1' with
the '/usr/local/cuda' symlink using the following two commands. Be
sure not to install the cuda drivers in this step.

    $ wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
    $ sudo sh cuda_10.1.243_418.87.00_linux.run

Commands found at https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

Create a new conda environment

    $ conda create --name pytorch
    $ conda activate pytorch

Install python 3.7

    $ conda install python=3.7

Install pytorch, torchvision, and cudatoolkit

    $ conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.1 -c pytorch

Install fvcore, cython and pycocotools

    $ pip install git+https://github.com/xdaimon/fvcore
    $ pip install cython==0.29.14
    $ pip install git+https://github.com/xdaimon/cocoapi.git#subdirectory=PythonAPI

Finally (might take 5-10 minutes),

    $ pip install git+https://github.com/xdaimon/detectron2.git
    $ pip install opencv-python==4.2.0.32 pyqt5==5.14.1

