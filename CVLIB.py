import dlib
import numpy as np
import cv2 as cv


def detect_face(imageAsMatrix):
	# Get the face detector
	faceDetector = dlib.get_frontal_face_detector()
	im = imageAsMatrix
	imDlib = cv.cvtColor(im, cv.COLOR_BGR2RGB)

	# Detect faces in the image
	faceRects = faceDetector(imDlib, 0)
	#print("Number of faces detected: ",len(faceRects))
	return faceRects

def get_landmarks(imageAsMatrix, faceRect):
	# Landmark model location
	PREDICTOR_PATH = "./1/data/models/shape_predictor_68_face_landmarks.dat"
	# The landmark detector is implemented in the shape_predictor class
	landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
	# List to store landmarks of all detected faces
	landmarksAll = []
	"""
	TODO
	if(len(faceRect) > 1):
		# Loop over all detected face rectangles
		for i in range(0, len(faceRects)):
			print("more than one face in image, implementation not finished")
			newRect = dlib.rectangle(int(faceRects[i].left()), int(faceRects[i].top()), int(faceRects[i].right()), int(faceRects[i].bottom()))
			# For every face rectangle, run landmarkDetector
			landmarks = landmarkDetector(imDlib, newRect)
			landmarksAll.append(landmarks)
			"""
	im = imageAsMatrix
	imDlib = cv.cvtColor(im, cv.COLOR_BGR2RGB)
	size = im.shape
	#rect = dlib.rectangle(int(faceRects[0].left()),int(faceRects[0].top()),int(faceRects[0].right()),int(faceRects[0].bottom()))
	landmarks = landmarkDetector(imDlib, faceRect)
	for p in landmarks.parts():
		landmarksAll.append((int(p.x),int(p.y)))
	return landmarksAll
	
# draw landmarks on image
def draw_landmarks(img, landmarks, withIndex=True):
	i = 0
	for point in landmarks:
		i = i+1
		#print(point)
		px,py = point
		# draw circle at each landmark
		cv.circle(img, (px, py), 1, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
		if(withIndex):
			# write landmark number at each landmark
			cv.putText(img, str(i+1), (px, py), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 100), 1)
	return img
	
def drawPolyline(image, landmarks, start, end, isClosed=False):
  points = []
  for i in range(start, end+1):
    point = [landmarks.part(i).x, landmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.polylines(im, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

# Use this function for 70-points facial landmark detector model
def renderFace(image, landmarks):
    assert(landmarks.num_parts == 68)
    drawPolyline(im, landmarks, 0, 16)           # Jaw line
    drawPolyline(im, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(im, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(im, landmarks, 27, 30)          # Nose bridge
    drawPolyline(im, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(im, landmarks, 36, 41, True)    # Left eye
    drawPolyline(im, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(im, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(im, landmarks, 60, 67, True)    # Inner lip
