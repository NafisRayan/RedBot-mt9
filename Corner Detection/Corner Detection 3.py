import cv2

# Initialize the camera capture
capture = cv2.VideoCapture(0)

# Function to change the Canny edge detection thresholds
def update_thresholds(_):
    global low_threshold, high_threshold
    low_threshold = cv2.getTrackbarPos('Low Threshold', 'Edge Detection')
    high_threshold = cv2.getTrackbarPos('High Threshold', 'Edge Detection')

# Create a window for the trackbars
cv2.namedWindow('Edge Detection')
cv2.createTrackbar('Low Threshold', 'Edge Detection', 50, 300, update_thresholds)
cv2.createTrackbar('High Threshold', 'Edge Detection', 150, 300, update_thresholds)

# Initialize threshold values
low_threshold = 50
high_threshold = 150

while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Use Canny edge detection with adjustable thresholds
    edges = cv2.Canny(blurred_frame, threshold1=low_threshold, threshold2=high_threshold)

    # Find and draw contours of detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    # Display the frame with detected edges
    cv2.imshow('Edge Detection', frame)

    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit the loop
        break

# Release the camera and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
