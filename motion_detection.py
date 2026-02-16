import cv2
import datetime
import csv

# Open camera
cam = cv2.VideoCapture(0)

# Read first two frames
ret, frame1 = cam.read()
ret, frame2 = cam.read()

# Motion log file
log_file = "motion_log.csv"

while cam.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion = False

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        motion = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)

    if motion:
        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"motion_images/{time_stamp}.jpg", frame1)
        cv2.putText(frame1, "MOTION SAVED", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Append log
        now = datetime.datetime.now()
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now.date(), now.time().strftime("%H:%M:%S"), "Motion Detected"])

    cv2.imshow("Motion Detection", frame1)

    frame1 = frame2
    ret, frame2 = cam.read()

    if cv2.waitKey(1) == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()

