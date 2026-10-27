import os

import cv2
from fer.fer import FER


# Use MTCNN to avoid the missing Haar cascade file error.
detector = FER(mtcnn=True)

video_capture = cv2.VideoCapture(0)

# Ask the webcam for a smaller frame to reduce processing.
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video_capture.isOpened():
    raise RuntimeError("Could not open the webcam.")


# Load duck.png from the same folder as this Python script.
script_folder = os.path.dirname(os.path.abspath(__file__))
duck_path = os.path.join(script_folder, "duck.png")
duck = cv2.imread(duck_path, cv2.IMREAD_UNCHANGED)

if duck is None:
    raise FileNotFoundError(
        f"Could not load {duck_path}. Put a transparent duck.png "
        "in the same folder as this script."
    )

duck = cv2.resize(duck, (80, 80))


def overlay_png(background, overlay, x, y):
    """Draw a transparent PNG over an OpenCV frame."""
    overlay_height, overlay_width = overlay.shape[:2]

    # Keep the image inside the frame.
    x = max(0, min(x, background.shape[1] - overlay_width))
    y = max(0, min(y, background.shape[0] - overlay_height))

    background_area = background[
        y : y + overlay_height,
        x : x + overlay_width
    ]

    if overlay.shape[2] == 4:
        # Use the PNG's alpha channel for transparency.
        alpha = overlay[:, :, 3:4].astype(float) / 255.0
        foreground = overlay[:, :, :3].astype(float)
        background_area_float = background_area.astype(float)

        blended = alpha * foreground + (1 - alpha) * background_area_float
        background[
            y : y + overlay_height,
            x : x + overlay_width
        ] = blended.astype("uint8")
    else:
        # Handle a PNG that has no transparency channel.
        background[
            y : y + overlay_height,
            x : x + overlay_width
        ] = overlay[:, :, :3]

    return background


# Speed and display settings.
DETECT_EVERY_N_FRAMES = 5
DETECTION_WIDTH = 320
MIN_CONFIDENCE = 0.40

frame_count = 0
latest_faces = []
current_emotion = "Waiting..."
duck_y = 130

print("Webcam active. Press 'q' in the video window to quit.")

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Camera reading failed.")
        break

    frame_count += 1
    height, width = frame.shape[:2]

    # Run emotion detection every few frames, using a smaller image.
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        detection_scale = width / DETECTION_WIDTH
        small_height = max(1, int(height / detection_scale))

        small_frame = cv2.resize(
            frame,
            (DETECTION_WIDTH, small_height)
        )

        latest_faces = detector.detect_emotions(small_frame)

        # Use the first detected face to control the duck.
        if latest_faces:
            emotions = latest_faces[0]["emotions"]
            current_emotion = max(emotions, key=emotions.get)
            confidence = emotions[current_emotion]

            if confidence >= MIN_CONFIDENCE:
                if current_emotion == "happy":
                    duck_y = max(90, duck_y - 35)       # Duck moves up
                elif current_emotion == "surprise":
                    duck_y = min(height - 90, duck_y + 35)  # Duck moves down
            else:
                current_emotion = "Uncertain"
        else:
            current_emotion = "No face detected"

    # Draw the most recent face detections.
    if latest_faces:
        detection_scale = width / DETECTION_WIDTH

        for face in latest_faces:
            x, y, box_width, box_height = face["box"]

            # Convert the face box to the original frame size.
            x = int(x * detection_scale)
            y = int(y * detection_scale)
            box_width = int(box_width * detection_scale)
            box_height = int(box_height * detection_scale)

            emotions = face["emotions"]
            emotion = max(emotions, key=emotions.get)
            confidence = emotions[emotion]

            if confidence >= MIN_CONFIDENCE:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_width, y + box_height),
                    (0, 255, 0),
                    2
                )

                label = f"{emotion.capitalize()} ({confidence:.0%})"
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Display the emotion label and duck in the top-right corner.
    label_x = max(10, width - 190)
    cv2.putText(
        frame,
        f"Emotion: {current_emotion.capitalize()}",
        (label_x, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    duck_x = width - 100
    frame = overlay_png(frame, duck, duck_x, duck_y)

    cv2.imshow("Duck Emotion Game", frame)

    # Press q while the video window is selected to quit.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()