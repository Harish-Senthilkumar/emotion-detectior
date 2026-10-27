import cv2
import random
import threading
import time
from queue import Queue, Empty, Full

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

# Use MTCNN to avoid the missing Haar cascade file error.
# Emotion detection runs in a separate thread to keep the camera preview responsive.
detector = FER(mtcnn=True)


# -------------------- Camera --------------------

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not video_capture.isOpened():
    raise RuntimeError("Could not open the webcam.")


# -------------------- Background emotion detection --------------------

# A queue of size one keeps only the newest frame waiting for analysis.
# This prevents a backlog of old frames from building up.
detection_queue = Queue(maxsize=1)
results_lock = threading.Lock()

latest_faces = []
latest_emotion = "Waiting..."
latest_confidence = 0.0
worker_running = True


def emotion_worker():
    """Analyze webcam frames in the background."""
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
                confidence = emotions[emotion]

            with results_lock:
                latest_faces = faces
                latest_emotion = emotion
                latest_confidence = confidence

        except Exception as error:
            print(f"Emotion detection error: {error}")


worker_thread = threading.Thread(target=emotion_worker, daemon=True)
worker_thread.start()


def send_latest_frame_for_detection(frame):
    """Queue a small frame, replacing an older waiting frame if necessary."""
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


# -------------------- Game helpers --------------------

def draw_duck(frame, center_x, center_y):
    """Draw a simple cartoon duck using OpenCV shapes."""
    # Body and head
    cv2.ellipse(
        frame,
        (center_x, center_y),
        (DUCK_RADIUS + 10, DUCK_RADIUS),
        0, 0, 360,
        (0, 220, 255),
        -1
    )
    cv2.circle(
        frame,
        (center_x + 17, center_y - 20),
        16,
        (0, 220, 255),
        -1
    )

    # Beak
    cv2.fillConvexPoly(
        frame,
        [
            (center_x + 29, center_y - 22),
            (center_x + 48, center_y - 16),
            (center_x + 29, center_y - 10),
        ],
        (0, 130, 255)
    )

    # Eye
    cv2.circle(
        frame,
        (center_x + 21, center_y - 24),
        3,
        (0, 0, 0),
        -1
    )

    # Wing
    cv2.ellipse(
        frame,
        (center_x - 5, center_y),
        (13, 8),
        -15, 0, 360,
        (0, 180, 230),
        -1
    )


def make_obstacle(frame_width, frame_height):
    """Create an obstacle at the right edge with a random vertical position."""
    margin = 55
    return {
        "x": frame_width + OBSTACLE_RADIUS,
        "y": random.randint(margin, frame_height - margin),
        "scored": False,
    }


# -------------------- Main game loop --------------------

frame_count = 0
duck_y = CAMERA_HEIGHT // 2
score = 0
game_over = False

obstacles = []
last_obstacle_time = time.time()
obstacle_interval = 1.5

fps = 0.0
last_fps_time = time.time()
fps_frame_count = 0

print("Game started. Press 'q' to quit. Press 'r' to restart after game over.")

try:
    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Camera reading failed.")
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        frame_count += 1
        fps_frame_count += 1

        # Submit a smaller frame for background emotion analysis.
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            detection_height = max(
                1,
                int(frame_height * DETECTION_WIDTH / frame_width)
            )
            small_frame = cv2.resize(
                frame,
                (DETECTION_WIDTH, detection_height)
            )
            send_latest_frame_for_detection(small_frame)

        # Read the most recent emotion result.
        with results_lock:
            faces = latest_faces
            emotion = latest_emotion
            confidence = latest_confidence

        # Use the detected expression as a game control.
        if confidence >= MIN_CONFIDENCE:
            if emotion == "happy":
                duck_y -= DUCK_MOVE_STEP
            elif emotion == "surprise":
                duck_y += DUCK_MOVE_STEP

        # Keep the duck inside the camera image.
        duck_y = max(DUCK_RADIUS + 10, min(frame_height - DUCK_RADIUS - 10, duck_y))

        # Game speed increases as the score rises.
        speed = min(
            MAX_SPEED,
            START_SPEED + score // SPEED_INCREASE_EVERY
        )

        # Spawn obstacles.
        now = time.time()
        if not game_over and now - last_obstacle_time >= obstacle_interval:
            obstacles.append(make_obstacle(frame_width, frame_height))
            last_obstacle_time = now

        # Move obstacles and check for collisions and points.
        if not game_over:
            for obstacle in obstacles:
                obstacle["x"] -= speed

                distance_x = abs(obstacle["x"] - DUCK_X)
                distance_y = abs(obstacle["y"] - duck_y)

                # Simple circular collision check.
                if (
                    distance_x < OBSTACLE_RADIUS + DUCK_RADIUS
                    and distance_y < OBSTACLE_RADIUS + DUCK_RADIUS
                ):
                    game_over = True

                # Earn a point when an obstacle passes the duck.
                if not obstacle["scored"] and obstacle["x"] < DUCK_X:
                    obstacle["scored"] = True
                    score += 1

            # Remove obstacles that have moved off screen.
            obstacles = [
                obstacle
                for obstacle in obstacles
                if obstacle["x"] > -OBSTACLE_RADIUS
            ]

        # Draw obstacles.
        for obstacle in obstacles:
            cv2.circle(
                frame,
                (int(obstacle["x"]), obstacle["y"]),
                OBSTACLE_RADIUS,
                (40, 40, 230),
                -1
            )
            cv2.circle(
                frame,
                (int(obstacle["x"] - 7), obstacle["y"] - 7),
                5,
                (120, 120, 255),
                -1
            )

        # Draw the duck and game information.
        draw_duck(frame, DUCK_X, duck_y)

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
            2
        )

        cv2.putText(
            frame,
            f"Score: {score}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Speed: {speed}",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Show approximate camera display FPS.
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = fps_frame_count / (current_time - last_fps_time)
            fps_frame_count = 0
            last_fps_time = current_time

        cv2.putText(
            frame,
            f"Camera FPS: {fps:.0f}",
            (frame_width - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if game_over:
            cv2.putText(
                frame,
                "GAME OVER",
                (frame_width // 2 - 150, frame_height // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
            cv2.putText(
                frame,
                "Press R to restart",
                (frame_width // 2 - 130, frame_height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
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