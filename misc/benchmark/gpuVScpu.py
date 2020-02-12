from tqdm import tqdm
import time
from Video import Video

#### How much faster is gpu processing vs cpu processing? ####

# Times to process an 11 second clip from the dataset
#      Detectron | YOLO
# GPU    01:45   | 00:22
# CPU  1:00:00   | 12:00
#
# in terms of fps reported by tqmd
#      Detectron | YOLO
# GPU     3.4    | 16
# CPU     1/12   | 2.1

#from VehicleDetectorDetectron import VehicleDetectorDetectron as Detector
from VehicleDetectorYolo import VehicleDetectorYolo as Detector
useGpu=False

video = Video('video/Gen5_RU_2019-10-07_07-56-42-0001_m0.avi')
detector = Detector(useGpu=useGpu)
for i in tqdm(range(video.getTotalNumFrames())):
    end, frame = video.getFrame(convRGB=detector.wantsRGB)
    features = detector.getFeatures(frame)
