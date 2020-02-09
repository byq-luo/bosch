This installation enables you to run object detection using YOLO only.
This will not enable semantic segmentation.
For that you'd need to follow the Linux or Windows intall guide.
Using detectron in this project is not supported on MacOS.

Run the following commands

    $ conda create --name pytorch
    $ conda activate pytorch
    $ conda install python=3.7
    $ conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.1 -c pytorch
    $ pip install opencv-python==4.2.0.32 pyqt5==5.14.1

Then download the YOLO weights from

    https://drive.google.com/open?id=1Jm8kqnMdMGUUxGo8zMFZMJ0eaPwLkxSG

and place them in

    {projectRootDir}/yolo/weights
