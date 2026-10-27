import random
import threading
import time
from queue import Empty, Full, Queue

import cv2
import numpy as np
from fer.fer import FER


# -------------------- Settings --------------------

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DETECTION_WIDTH = 320
DETECT_EVERY_N_FRAMES = 5
MIN_CONFIDENCE = 0.40

DUCK_X = 120
DUCK_RADIUS = 24
DUCK_MOVE_STEP = 35

OBSTACLE_RADIUS = 22
START_SPEED = 4
MAX_SPEED = 14
SPEED_INCREASE_EVERY = 5


# MTCNN avoids the missing Haar cascade file error.
detector = FER(mtcnn=True)


# -------------------- Camera --------------------

# CAP_DSHOW can help webcam startup and responsiveness on Windows.
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    # Try OpenCV's default camera backend if DirectShow did not open it.
    video_capture.release()
    video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not video_capture.isOpened():
    raise RuntimeError("Could not open the webcam. Check that it is connected and not in use.")


# -------------------- Background emotion detection --------------------

# Keep at most one waiting frame, so old frames do not build up in a queue.
detection_queue = Queue(maxsize=1)
results_lock = threading.Lock()

latest_faces = []
latest_emotion = "Waiting..."
latest_confidence = 0.0
worker_running = True


def emotion_worker():
    """Analyze frames in the background so detection does not block the camera preview."""
    global latest_faces
    global latest_emotion
    global latest_confidence

    while worker_running:
        try:
            small_frame = detection_queue.get(timeout=0.1)
        except Empty:
            continue

        if small_frame is None:
            break

        try:
            faces = detector.detect_emotions(small_frame)

            emotion = "No face"
            confidence = 0.0

            if faces:
                emotions = faces[0]["emotions"]
                emotion = max(emotions, key=emotions.get)
                confidence = float(emotions[emotion])

            with results_lock:
                latest_faces = faces
                latest_emotion = emotion
                latest_confidence = confidence

        except Exception as error:
            print(f"Emotion detection error: {error}")


worker_thread = threading.Thread(target=emotion_worker, daemon=True)
worker_thread.start()


def send_latest_frame_for_detection(frame):
    """Queue the newest frame and discard an older waiting frame if needed."""
    try:
        detection_queue.put_nowait(frame)
    except Full:
        try:
            detection_queue.get_nowait()
        except Empty:
            pass

        try:
            detection_queue.put_nowait(frame)
        except Full:
            pass


# -------------------- Game drawing helpers --------------------

def draw_duck(frame, center_x, center_y):
    """Draw a simple cartoon duck using OpenCV."""
    # Body
    cv2.ellipse(
        frame,
        (center_x, center_y),
        (DUCK_RADIUS + 10, DUCK_RADIUS),
        0,
        0,
        360,
        (0, 220, 255),
        -1,
    )

    # Head
    cv2.circle(
        frame,
        (center_x + 17, center_y - 20),
        16,
        (0, 220, 255),
        -1,
    )

    # Beak: OpenCV expects the polygon points as a NumPy array.
    beak_points = np.array(
        [
            [center_x + 29, center_y - 22],
            [center_x + 48, center_y - 16],
            [center_x + 29, center_y - 10],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, beak_points, (0, 130, 255))

    # Eye
    cv2.circle(
        frame,
        (center_x + 21, center_y - 24),
        3,
        (0, 0, 0),
        -1,
    )

    # Wing
    cv2.ellipse(
        frame,
        (center_x - 5, center_y),
        (13, 8),
        -15,
        0,
        360,
        (0, 180, 230),
        -1,
    )


def make_obstacle(frame_width, frame_height):
    """Create an obstacle at the right side with a random vertical position."""
    margin = OBSTACLE_RADIUS + DUCK_RADIUS + 10
    safe_height = max(margin, frame_height - margin)

    return {
        "x": frame_width + OBSTACLE_RADIUS,
        "y": random.randint(margin, safe_height),
        "scored": False,
    }


# -------------------- Game state --------------------

frame_count = 0
duck_y = CAMERA_HEIGHT // 2
score = 0
game_over = False

obstacles = []
last_obstacle_time = time.time()
obstacle_interval = 1.5

fps = 0.0
fps_frame_count = 0
last_fps_time = time.time()


# -------------------- Main loop --------------------

print("Game started. Press 'q' to quit.")
print("Press 'r' to restart after game over.")

try:
    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Camera reading failed.")
            break

        # Mirror the image so movement feels natural.
        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]
        frame_count += 1
        fps_frame_count += 1

        # Submit a smaller image to the emotion detector every few frames.
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            detection_height = max(
                1,
                int(frame_height * DETECTION_WIDTH / frame_width),
            )

            small_frame = cv2.resize(
                frame,
                (DETECTION_WIDTH, detection_height),
                interpolation=cv2.INTER_AREA,
            )

            send_latest_frame_for_detection(small_frame)

        # Copy the latest detection result safely from the worker thread.
        with results_lock:
            faces = latest_faces
            emotion = latest_emotion
            confidence = latest_confidence

        # Facial expressions control the duck:
        # happy moves up, surprise moves down.
        if not game_over and confidence >= MIN_CONFIDENCE:
            if emotion == "happy":
                duck_y -= DUCK_MOVE_STEP
            elif emotion == "surprise":
                duck_y += DUCK_MOVE_STEP

        # Keep the duck inside the visible camera frame.
        duck_y = max(
            DUCK_RADIUS + 10,
            min(frame_height - DUCK_RADIUS - 10, duck_y),
        )

        # Increase game speed by one for every five points.
        speed = min(
            MAX_SPEED,
            START_SPEED + score // SPEED_INCREASE_EVERY,
        )

        # Add a new obstacle periodically.
        now = time.time()
        if not game_over and now - last_obstacle_time >= obstacle_interval:
            obstacles.append(make_obstacle(frame_width, frame_height))
            last_obstacle_time = now

        # Move obstacles and check collisions.
        if not game_over:
            for obstacle in obstacles:
                obstacle["x"] -= speed

                distance_x = abs(obstacle["x"] - DUCK_X)
                distance_y = abs(obstacle["y"] - duck_y)

                # Approximate collision check between the duck and obstacle.
                if (
                    distance_x < OBSTACLE_RADIUS + DUCK_RADIUS
                    and distance_y < OBSTACLE_RADIUS + DUCK_RADIUS
                ):
                    game_over = True

                # Award one point when an obstacle passes the duck.
                if not obstacle["scored"] and obstacle["x"] < DUCK_X:
                    obstacle["scored"] = True
                    score += 1

            # Remove obstacles that have left the screen.
            obstacles = [
                obstacle
                for obstacle in obstacles
                if obstacle["x"] > -OBSTACLE_RADIUS
            ]

        # Draw obstacles.
        for obstacle in obstacles:
            center = (int(obstacle["x"]), int(obstacle["y"]))

            cv2.circle(
                frame,
                center,
                OBSTACLE_RADIUS,
                (40, 40, 230),
                -1,
            )
            cv2.circle(
                frame,
                (center[0] - 7, center[1] - 7),
                5,
                (120, 120, 255),
                -1,
            )

        # Draw the duck.
        draw_duck(frame, DUCK_X, duck_y)

        # Draw the current expression.
        emotion_text = emotion.capitalize()
        if confidence >= MIN_CONFIDENCE:
            emotion_text += f" ({confidence:.0%})"

        cv2.putText(
            frame,
            f"Expression: {emotion_text}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        # Draw score and speed.
        cv2.putText(
            frame,
            f"Score: {score}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Speed: {speed}",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # Update and display approximate camera FPS.
        current_time = time.time()
        elapsed = current_time - last_fps_time

        if elapsed >= 1.0:
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            last_fps_time = current_time

        cv2.putText(
            frame,
            f"Camera FPS: {fps:.0f}",
            (max(10, frame_width - 180), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Game-over screen.
        if game_over:
            cv2.putText(
                frame,
                "GAME OVER",
                (max(10, frame_width // 2 - 150), frame_height // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                "Press R to restart",
                (max(10, frame_width // 2 - 130), frame_height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Duck Emotion Game", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r") and game_over:
            score = 0
            duck_y = frame_height // 2
            obstacles = []
            game_over = False
            last_obstacle_time = time.time()


finally:
    worker_running = False

    try:
        detection_queue.put_nowait(None)
    except Full:
        pass

    worker_thread.join(timeout=2)
    video_capture.release()
    cv2.destroyAllWindows()