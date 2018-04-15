# Mobius

Mobius is designed to be an mobile app that will reward good driving behavior with insurance discounts. Our current version is a PC-based solution in "driver-detection-system" forder that simulate the real-time app behavior. The actual app is still in progess and will be released in the future.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Our program is based on the Python 3.6.1. Our program has been tested on the Ubuntu16.04 and Mac OS 10. Here, is the library you need to install and how to install them.

* [dlib](https://github.com/davisking/dlib) - with python API
* imutils
* opencv 3.4.1

### Installing

Our program use the pre-trained models to detect face features. Runing the command before to download the required pre-trained model.

```
cd /path/to/targetfolder
wget  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
tar xvjf shape_predictor_68_face_landmarks.dat.bz2
```

Also for security purpose, our app is only bind with one users. So user should replace the "mypic.jpg" with your own picture before running the live demo.

## Running the tests

To run the test, simply call

```
cd driver-detection-system/
python3 dowsiness_and_distraction_detect.py
```

Basically, this program takes in each frame of video streaming. Detect the bounding box for face and call the following functions to do the further analysis on the face feature.

<img src="https://github.com/HarryisCoder/Mobius/blob/master/demo_results/FaceRecog_failed.PNG" width="400"/> <img src="https://github.com/HarryisCoder/Mobius/blob/master/demo_results/FaceRecog_success.png" width="400"/>

### Face Recognition

This can encode a face image into a feature vector. So each time we calculate the distance between the vector generated by the take-in frame and the vector generated by the profile picture. If the difference is lower than certain threshold, the identity matches.


<img src="https://github.com/HarryisCoder/Mobius/blob/master/demo_results/drowsiness_and_distraction_Drowsy!.png" width="400"/> <img src="https://github.com/HarryisCoder/Mobius/blob/master/demo_results/drowsiness_and_distraction_distract!.png" width="400"/> <img src="https://github.com/HarryisCoder/Mobius/blob/master/demo_results/drowsiness_and_distraction_normal.png" width="400"/>

### Drowsiness Detection
This can detect the eye size changing and can thus detect behaviors like closing eyes or fast winkle, which shows the drowsiness level. In the demo, the eye size changing is called "Eye Aspect Ratio". 

### Distraction Detection
This can calculate the relative head rotation versus the fixed x-axis, y-axis and z-axis. By comparing the head pose with standard head direction, we can analyze some behaviors like texting or talking to the person next to your. In the demo, the rotation is represented as "head: x: y: z: ".

## Explaination on other files
The "face recognition folder" contains some related functions for face recognition. The 'alarm.wav' is the sound file for warning user's bad driving behavior. 


## Authors

* **TianShu Cheng** - [HarryisCoder](https://github.com/HarryisCoder)
	- *Identity recognition*
	- *Drowsiness detection*
	- *General framework*
* **Yifang Chen** - [cloudwaysX](https://github.com/cloudwaysX)
	- *Distraction (head pose) detection*
	- *General framework*




## Acknowledgments

We refered the following code in our project.

* [Learn OpenCV: Head pose estimtion](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)

* [pyImagesearch:Drowsiness detection](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)

* [pypi: Face Recognition](https://pypi.python.org/pypi/face_recognition)


