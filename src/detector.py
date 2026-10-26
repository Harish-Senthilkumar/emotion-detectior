'''
• import cv2: Imports OpenCV (Open Source Computer Vision Library). 
This library handles pixels, opens your webcam, reads raw video data, and draws shapes directly on your screen.

• from fer import FER: Imports the Facial Emotion Recognition (FER) class from the fer library. 
This is the pre-trained artificial intelligence model that looks at facial features and classifies expressions.

'''
# Import main libaries
import cv2 # For camera and video handling (pixels)
from fer.fer import FER # pre-trained model & identifies facial features

detector = FER(mtcnn=True) # Initialize the AI model; mtcnn=False makes it faster for real-time video streaming

# Lets python connect to our hardware's default camera
video_capture = cv2.VideoCapture(0)

print("Webcam feed live and active. Press 'q' inside the video windwo to quit!")

# Infinite loop to process the live camera frame-by-frame
while True:
    # Capture an instantaneuous snapshot image array (frame) from the webcam link
    # ret is a True/False flag showing if the camera frame was read successfully
    ret, frame = video_capture.read()

    if not ret:
        # printing an error notification if camera failed
        print("Camera reading failed.")
        break # out the loop
    
    # Pass the current snapshot image frame array to the AI model for face analysis
    analysis  = detector.detect_emotions(frame) 

# Loop through all singular faces discovered by the AI-model within the frame
    for face in analysis:
        # Unpack the top-left coordinate (x,y) along with the width (w) and height (h)
        x, y, w, h = face["box"]
        # Extract the string name of the emotion with the highest probability score
        emotion = max(face["emotions"], key=face["emotions"].get)
        # Look up the exact floating-point percentage score for that dominant emotion
        confidence = face["emotions"][emotion]

        # Draw a bright green rectangle border directly matching the face boundaries
        # (0, 255, 0) represents Blue=0, Green=255, Red=0 in BGR color space format
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0, 255, 0), 2)

        # Capitalizing the emotion strign and format the decimal score into a clean integer percentage
        label = f"{emotion.capitalize()} ({int(confidence * 100)}%)"
        # Paint the text lavel onto the frame array 10 pixels directly above the rectangular box

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # Making a clean user desktop pop-up interface screen to display the frame array
    cv2.imshow("Live Emotion Detector", frame)

# Pause execution 1ms to look for keyboard inputs from the user
    # If the user presses the 'q' key down on their keyboard, terminate the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break # break out of the tracking infinite-loop


# Shut down the active camera lens connection and turn off the webcame indector hardware light
video_capture.release()
# Close and completely wipe out all windows opened by OpenCV
cv2.destroyAllWindows()