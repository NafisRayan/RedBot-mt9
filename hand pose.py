import cv2
import mediapipe as mp

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera (you can use a video stream or an image)
cap = cv2.VideoCapture(0)  # Use the default camera (change to your camera source if needed)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the Z-coordinate (depth) of a specific landmark, e.g., the thumb tip (change as needed)
            thumb_tip_depth = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

            # You can use thumb_tip_depth to estimate the distance in a real-world unit (e.g., centimeters)
            # You may need to calibrate this based on your camera and setup.

            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
