import cv2

# Start video capture (0 = webcam)
cap = cv2.VideoCapture(0)

# Read first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

    # Find difference between frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold to get motion areas
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:

        # Ignore small movements
        if cv2.contourArea(contour) < 1000:
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle around motion
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame1, "Motion Detected", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Show frame
    cv2.imshow("Motion Detection", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:   # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()