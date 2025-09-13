import pygame
import cv2
import numpy as np
import sys
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize Pygame
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("MediaPipe Hand Tracking")

font = pygame.font.SysFont(None, 48)
clock = pygame.time.Clock()

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    result = hands.process(rgb_frame)

    # Convert frame to Pygame surface
    rgb_frame = np.rot90(rgb_frame)  # Rotate for Pygame orientation
    frame_surface = pygame.surfarray.make_surface(rgb_frame)
    screen.blit(frame_surface, (0, 0))

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = int(lm.x * screen_width)
                y = int(lm.y * screen_height)
                pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)

    # Optional: Overlay text
    text = font.render("Recording...", True, (255, 0, 0))
    screen.blit(text, (20, 20))

    # Refresh screen
    pygame.display.flip()
    clock.tick(30)

    # Pygame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Cleanup
cap.release()
pygame.quit()
hands.close()