import cv2
import datetime
import csv
import threading
import tkinter as tk
from tkinter import messagebox

# ------------------------
# Configuration
# ------------------------
log_file = "motion_log.csv"  # CSV log file
image_folder = "motion_images"  # Folder to save images

# Create folder if it doesn't exist
import os
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# ------------------------
# Motion Detection Function
# ------------------------
def start_motion_detection():
    cam = cv2.VideoCapture(0)
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    while running[0]:
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
            cv2.imwrite(f"{image_folder}/{time_stamp}.jpg", frame1)
            now = datetime.datetime.now()
            # Log to CSV
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([now.date(), now.time().strftime("%H:%M:%S"), "Motion Detected"])
            # Console message
            print(f"[{now.strftime('%H:%M:%S')}] Motion detected! Image saved: {image_folder}/{time_stamp}.jpg")
            # Update GUI label
            status_label.config(text=f"Motion detected! Image saved: {time_stamp}.jpg")

        cv2.imshow("Motion Detection", frame1)

        frame1 = frame2
        ret, frame2 = cam.read()

        if cv2.waitKey(1) == 27 or not running[0]:
            break

    cam.release()
    cv2.destroyAllWindows()
    status_label.config(text="Detection stopped")

# ------------------------
# GUI Setup
# ------------------------
def start():
    if not running[0]:
        running[0] = True
        threading.Thread(target=start_motion_detection).start()
        status_label.config(text="Detection started...")

def stop():
    running[0] = False
    messagebox.showinfo("Motion Detection", "Detection Stopped")

running = [False]

root = tk.Tk()
root.title("Motion Detection System")
root.geometry("400x200")

# Buttons
start_btn = tk.Button(root, text="Start Detection", command=start, width=25, bg="green", fg="white")
start_btn.pack(pady=20)

stop_btn = tk.Button(root, text="Stop Detection", command=stop, width=25, bg="red", fg="white")
stop_btn.pack(pady=10)

# Status Label
status_label = tk.Label(root, text="Status: Waiting...", fg="blue")
status_label.pack(pady=5)

root.mainloop()

