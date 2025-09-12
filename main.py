import pygame
import cv2
import numpy as np
import sys

# Initialize Pygame
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Motion Recorder")

font = pygame.font.SysFont(None, 48)
clock = pygame.time.Clock()

# Initialize OpenCV Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

# Set camera resolution to match the screen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Motion detection: initial frame
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

def detect_motion(prev, current, threshold=1000):
    frame_delta = cv2.absdiff(prev, current)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    motion_pixels = cv2.countNonZero(thresh)
    return motion_pixels > threshold

running = True
while running:
    ret, frame2 = cap.read()
    if not ret:
        break

    # Prepare for motion detection
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)
    motion_detected = detect_motion(frame1_gray, frame2_gray)
    frame1_gray = frame2_gray

    # Convert BGR to RGB for Pygame
    frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    frame_rgb = np.rot90(frame_rgb)  # Rotate because OpenCV and Pygame use different orientations
    frame_surface = pygame.surfarray.make_surface(frame_rgb)

    # Draw the frame
    screen.blit(frame_surface, (0, 0))

    # Overlay text if motion detected
    if motion_detected:
        text = font.render("Recording...", True, (255, 0, 0))
        screen.blit(text, (20, 20))

    pygame.display.flip()
    clock.tick(30)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Clean up
cap.release()
pygame.quit()