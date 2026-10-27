import cv2
from fer import FER

# Faster than MTCNN on many computers, though face detection may be less precise.
detector = FER(mtcnn=False)

video_capture = cv2.VideoCapture(0)

# Ask the camera for a smaller frame to reduce processing.
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video_capture.isOpened():
    raise RuntimeError("Could not open the webcam.")

print("Webcam active. Press 'q' in the video window to quit.")

DETECT_EVERY_N_FRAMES = 5
DETECTION_WIDTH = 320
MIN_CONFIDENCE = 0.40

frame_count = 0
latest_faces = []
scale = 1.0

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Camera reading failed.")
        break

    frame_count += 1

    # Run the AI model only every few frames.
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        height, width = frame.shape[:2]
        scale = width / DETECTION_WIDTH
        small_frame = cv2.resize(
            frame,
            (DETECTION_WIDTH, int(height / scale))
        )
        latest_faces = detector.detect_emotions(small_frame)

    # Draw the most recent detections on each displayed frame.
    for face in latest_faces:
        x, y, w, h = face["box"]

        # Convert coordinates from the smaller detection image to the full frame.
        x = int(x * scale)
        y = int(y * scale)
        w = int(w * scale)
        h = int(h * scale)

        emotions = face["emotions"]
        emotion = max(emotions, key=emotions.get)
        confidence = emotions[emotion]

        if confidence < MIN_CONFIDENCE:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{emotion.capitalize()} ({confidence:.0%})"
        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Live Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()