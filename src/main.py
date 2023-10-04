import argparse
from detector import *

# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Object Detection")
parser.add_argument("--camera", action="store_true", help="Use webcam as video source")
args = parser.parse_args()

# Set the video path based on the command-line argument
if args.camera:
    videoPath = 0
else:
    videoPath = "../test/street.mp4"

classFile = "coco.names"
imagePath = "../test/cats.jpg"
folderPath = "../test/"
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImagesInFolder(folderPath, threshold)
detector.predictImage(imagePath, threshold)

detector.predictVideo(videoPath, threshold)
