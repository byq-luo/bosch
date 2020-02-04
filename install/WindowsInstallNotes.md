Install Visual Studio Community 2019.
Make sure Miniconda3 is installed at C:\Miniconda3

Download & install CUDA

Do a custom install and make sure the only checked option is the Development components

![](image.png)

http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe

Now create a new conda environment

    $ conda create --name pytorch

Then activate the environment

    $ conda activate pytorch

Install python 3.7

    $ conda install python=3.7

Install pytorch, torchvision, and cudatoolkit=10.1

    $ conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.1 -c pytorch

Install fvcore, cython and pycocotools

    $ pip install git+https://github.com/xdaimon/fvcore
    $ pip install cython==0.29.14
    $ pip install git+https://github.com/xdaimon/cocoapi.git#subdirectory=PythonAPI

Change the file

    C:\Miniconda3\envs\pytorch\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h(190)
      static constexpr size_t DEPTH_LIMIT = 128;
        change to -->
      static const size_t DEPTH_LIMIT = 128;

And

    C:\Miniconda3\envs\pytorch\Lib\site-packages\torch\include\pybind11\cast.h(1449)
      explicit operator type&() { return *(this->value); }
        change to -->
      explicit operator type&() { return *((type*)this->value); }

Then execute

    $ "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

Then run the following commands (might take 5-20 minutes)

    $ conda activate pytorch
    $ pip install git+https://github.com/xdaimon/detectron2.git

Finally

    $ pip install opencv-python==4.2.0.32
    $ pip install pyqt5==5.14.1


