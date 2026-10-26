'''
• import cv2: Imports OpenCV (Open Source Computer Vision Library). 
This library handles pixels, opens your webcam, reads raw video data, and draws shapes directly on your screen.

• from fer import FER: Imports the Facial Emotion Recognition (FER) class from the fer library. 
This is the pre-trained artificial intelligence model that looks at facial features and classifies expressions.

'''
# Import main libaries
import cv2 # For camera and video handling (pixels)
from fer import FER # pre-trained model & identifies facial features

detector = FER(mtcnn=False) #Initialize the AI model; mtcnn=False makes it faster for real-time video streaming

 #Lets python connect to our hardware's default camera
video_capture = cv2.VideoCapture(0)

print("Webcam feed live and active. Press 'q' inside the video windwo to quit!")

# Infinite loop to process the live camera frame-by-frame
while True:
    # Capture an instantaneuous snapshot image array (frame) from the webcam link
    # ret is a True/False flag showing if the camera frame was read successfully
    ret, frame = video_capture.read()

    if not ret:
        #printing an error notification if camera failed
        print("Camera reading failed.")
        break # out the loop
    
    # Pass the current snapshot image frame array to the AI model for face analysis
    analysis  = detector.detect_emotions(frame) 

    