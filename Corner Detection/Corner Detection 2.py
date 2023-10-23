import cv2
import numpy as np

# Initialize the camera capture
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Use Canny edge detection to detect edges
    edges = cv2.Canny(blurred_frame, threshold1=30, threshold2=100)

    # Find and draw contours of detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store distances
    distances = []

    for contour in contours:
        # Calculate the center of each contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Calculate the distance from the center of the frame (image center)
        frame_height, frame_width = frame.shape[:2]
        distance = np.sqrt((cX - frame_width // 2) ** 2 + (cY - frame_height // 2) ** 2)

        distances.append(distance)

    max_distance = max(distances)

    for i in range(len(contours)):
        if distances[i] < max_distance * 0.5:  # Adjust the threshold as needed
            cv2.drawContours(frame, contours, i, (0, 0, 255), 2)  # Nearby edges (red)
        else:
            cv2.drawContours(frame, contours, i, (0, 255, 0), 2)  # Distant edges (green)

    # Display the frame with detected edges
    cv2.imshow('Edge Detection', frame)

    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit the loop
        break

# Release the camera and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
