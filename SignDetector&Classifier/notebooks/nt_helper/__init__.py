import sys
import os

print("helper init success")
YOLO_HOME_DIR = os.path.dirname(os.path.realpath(__file__)) + "//YoloV5"
sys.path.append(YOLO_HOME_DIR)
