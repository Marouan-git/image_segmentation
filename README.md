# Image segmentation

The purpose of this project is to provide an introduction to image segmentation. You'll find:
1. A notebook that provides an introduction to image segmentation using the  [DeepLabV3](https://pytorch.org/vision/main/models/deeplabv3.html) model and the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) : [intro_to_img_seg.ipynb](./intro_to_img_seg.ipynb).
2. A notebook with a real time image segmentation and quantization implemented using a pretrained [YOLOv11 model](https://docs.ultralytics.com/models/yolo11/#overview) from [Ultralytics](https://docs.ultralytics.com/) : [realtime_segmentation.ipynb](./realtime_segmentation.ipynb).
3. A Flutter mobile application which uses a YOLOv11 model for real time image segmentation: [image_segmentation_app](./image_segmentation_app/).

The [calibration_img folder](./calibration_img/) contains images used for static quantization.

## Install the environment for the notebooks

First clone the project:
```
git clone https://github.com/Marouan-git/image_segmentation.git
cd image_segmentation
```
Install pipenv if not already installed:
```
pip install pipenv
```

Install the virtual environment (dependencies are listed in the Pipfile and the Pipfile.lock):
```
pipenv install
```

Activate the virtual environment:
```
pipenv shell
```

## Flutter application

You need to have installed flutter and configure your emulator, see [Flutter docs](https://docs.flutter.dev/get-started/install).  
  
If you use your own smartphone, activate the Developer mode and enable the USB debugging in the Developer options.  
Once plugged to your computer, select your smartphone as device emulator and run the main.dart file.  
  
Currently, the segmentation is not smooth: the preprocessing and the postprocessing must be improved (model's inference time is OK).