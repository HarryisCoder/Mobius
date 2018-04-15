# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
# import loadOpenFace
import face_recognition

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# Camera internals
def calcRotation(im,image_points):
    size = im.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    # print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    # print( "Translation Vector:\n {0}".format(translation_vector))


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose


    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    # misc.imshow(cv2.line(im, p1, p2, (255,0,0), 2))
    return {'p1':p1,'p2':p2,'rotVec':rotation_vector}

# run detection program
def runDetection(gray):
	# define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold for to set off the
	# alarm
	EYE_AR_THRESH = 0.25
	EYE_AR_CONSEC_FRAMES = 24

	HEAD_THRESH_LEFTRIGHT = 0.25
	HEAD_THRESH_DOWN = 0.4
	HEAD_CONSEC_FRAMES = 48

	# initialize the frame counter as well as a boolean used to
	# indicate if the alarm is going off
	ALARM_ON = False

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# print("rects", len(rects))
	if (len(rects) == 0):
		cv2.putText(frame, "No face detected", (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		NOFACE_COUNTER += 1
		if (NOFACE_COUNTER > 24):
			cv2.putText(frame, "DISTRACTION ALERT!", (0, 120),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	else:
		NOFACE_COUNTER = 0
		rect = rects[0]

	# loop over the face detections
	# for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# print ("shape", shape.shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			# print("ear = ", ear)
			# print("ear is low!")
			global BLINK_COUNTER
			BLINK_COUNTER += 1
			# print("BLINK COUNTER: ", BLINK_COUNTER)

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=("./alarm.wav",))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (0, 90),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				global DROWSINESS_ALERT_FLAG
				if (DROWSINESS_ALERT_FLAG == False):
					global DROWSINESS_ALERT_COUNT
					DROWSINESS_ALERT_COUNT += 1
					# global DROWSINESS_ALERT_FLAG
					DROWSINESS_ALERT_FLAG == True
					print("DROWSINESS_ALERT_COUNT: ", DROWSINESS_ALERT_COUNT)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			BLINK_COUNTER = 0
			ALARM_ON = False
			# global DROWSINESS_ALERT_FLAG
			DROWSINESS_ALERT_FLAG = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear), (0, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		#calculate rotation
		image_points = np.array([
			            shape[30,:],          # nose tip
                        shape[8,:],           # Chin
                        shape[26,:],          # Left eye left corner
                        shape[17,:],          # Right eye right corne
                        shape[54,:],          # Left Mouth corner
                        shape[48,:]           # Right mouth corner
                        ], dtype="double")
		distractResult = calcRotation(frame,image_points)
		cv2.line(frame, distractResult['p1'], distractResult['p2'],(255,0,0), 2)
		cv2.putText(frame, "Head Rot.: x{} y{} z{}".format(distractResult['rotVec'][0], distractResult['rotVec'][1], distractResult['rotVec'][2]), (0, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		if abs(distractResult['rotVec'][0]) > HEAD_THRESH_LEFTRIGHT or \
		   abs(distractResult['rotVec'][2]) > HEAD_THRESH_LEFTRIGHT or \
		   abs(distractResult['rotVec'][1]) > HEAD_THRESH_DOWN:
			# print("ear = ", ear)
			# print("ear is low!")
			global HEAD_COUNTER
			HEAD_COUNTER += 1
			# print("BLINK COUNTER: ", BLINK_COUNTER)

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if HEAD_COUNTER >= HEAD_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DISTRACTION ALERT!", (0, 120),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				global DISTRACTION_ALERT_FLAG
				if (DISTRACTION_ALERT_FLAG == False):
					global DISTRACTION_ALERT_COUNT
					DISTRACTION_ALERT_COUNT += 1
					# global DISTRACTION_ALERT_FLAG
					DISTRACTION_ALERT_FLAG == True
					print("DISTRACTION_ALERT_COUNT: ", DISTRACTION_ALERT_COUNT)


		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			HEAD_COUNTER = 0
			ALARM_ON = False
			# global DISTRACTION_ALERT_FLAG
			DISTRACTION_ALERT_FLAG = False


################ main program #########################
# 3D model points.
model_points = np.array([(0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corne
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default = "shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default= "alarm.wav",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# parameter setting
FACE_ID_COUNTER = 0
BLINK_COUNTER = 0
NOFACE_COUNTER = 0
DROWSINESS_ALERT_COUNT = 0
DISTRACTION_ALERT_COUNT = 0
DROWSINESS_ALERT_FLAG = False
DISTRACTION_ALERT_FLAG = False

picture_of_me = face_recognition.load_image_file("mypic.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

while True:

	# key = cv2.waitKey(1) & 0xFF
	 
	# if the `q` key was pressed, break from the loop
	# if key == ord("e"):
	# 	break

	# login
	while True:
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=650)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# unknown_picture = face_recognition.load_image_file(frame)
		unknown_faces_encoding = face_recognition.face_encodings(frame)

		# Now we can see the two face encodings are of the same person with `compare_faces`!


		# mypic = cv2.imread("mypic.jpg");
		# print("mypic:", mypic.shape)
		# # face identification to log in
		# if (loadOpenFace.FaceRecognition(frame, mypic) != -1):

		if FACE_ID_COUNTER >= 20:
			if FACE_ID_COUNTER >= 25:
				break
			FACE_ID_COUNTER += 1
			print("FACE_ID_COUNTER:", FACE_ID_COUNTER)
			cv2.putText(frame, "Login successfully!", (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# show the frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			continue

		if len(unknown_faces_encoding) == 0:
			cv2.putText(frame, "You are not the right person, please log in again!", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			FACE_ID_COUNTER = 0
		else:
			unknown_face_encoding = unknown_faces_encoding[0]
			results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
			print("result:", results[0])
			if len(results) != 0 and results[0] == True:
				FACE_ID_COUNTER += 1
				print("FACE_ID_COUNTER:", FACE_ID_COUNTER)
				cv2.putText(frame, "Hi Tianshu, you are detected!", (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				FACE_ID_COUNTER = 0
				cv2.putText(frame, "You are not the right person, please log in again!", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		
		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if FACE_ID_COUNTER >= 25:
			break
	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
			# time.sleep(1)
		if key == ord("e"):
			exit(0)

	FACE_ID_COUNTER = 0
	# start app

	# loop over frames from the video stream
	while True:
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=650)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# run drowsiness & distraction detection for each frame
		runDetection(gray)

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		if key == ord("e"):
			exit(0)
	BLINK_COUNTER = 0
	NOFACE_COUNTER = 0

	while True:
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=650)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# run drowsiness & distraction detection for each frame
		# runDetection(gray)
		cv2.putText(frame, "DROWSINESS COUNT: " + str(DROWSINESS_ALERT_COUNT), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "DISTRACTION COUNT: " + str(DISTRACTION_ALERT_COUNT), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		if key == ord("e"):
			exit(0)
	DROWSINESS_ALERT_COUNT = 0
	DISTRACTION_ALERT_COUNT = 0
	DROWSINESS_ALERT_FLAG = False
	DISTRACTION_ALERT_FLAG = False

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
