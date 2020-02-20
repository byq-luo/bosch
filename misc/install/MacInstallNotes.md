This installation enables you to run object detection using YOLO only.
This will not enable semantic segmentation.
For that you'd need to follow the Linux or Windows intall guide.
Using detectron in this project is not supported on MacOS.

Run the following commands

    $ conda create --name bosch
    $ conda activate bosch
    $ conda install python=3.7
    $ conda install pytorch=1.4 torchvision=0.5 cudatoolkit=9.0 -c pytorch
    $ conda install pyqt opencv
