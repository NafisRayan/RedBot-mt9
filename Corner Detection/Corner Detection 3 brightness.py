import cv2
import numpy as np

# Initialize the camera capture
capture = cv2.VideoCapture(0)

# Function to apply color grading
def apply_color_grading(frame, brightness, contrast):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Adjust brightness and contrast
    frame_hsv[:, :, 2] = np.clip(contrast * frame_hsv[:, :, 2] + brightness, 0, 255)
    
    return cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

# Function to update color grading parameters
def update_color_grading(_):
    global brightness, contrast
    brightness = cv2.getTrackbarPos('Brightness', 'Color Grading') - 100
    contrast = cv2.getTrackbarPos('Contrast', 'Color Grading') / 100.0

# Create a window for color grading trackbars
cv2.namedWindow('Color Grading')
cv2.createTrackbar('Brightness', 'Color Grading', 100, 200, update_color_grading)
cv2.createTrackbar('Contrast', 'Color Grading', 100, 200, update_color_grading)

# Initialize color grading parameters
brightness = 0
contrast = 1.0

while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        break

    # Apply color grading
    graded_frame = apply_color_grading(frame, brightness, contrast)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(graded_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Use Canny edge detection with adjustable thresholds
    edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)

    # Find and draw contours of detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(graded_frame, contours, -1, (0, 0, 255), 2)

    # Display the frame with color grading and detected edges
    cv2.imshow('Edge Detection with Color Grading', graded_frame)

    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit the loop
        break

# Release the camera and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
