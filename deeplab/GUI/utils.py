import Tkinter
from load_img import MyLoadImage
import cv2
import numpy as np


def display_image(img_path, frame, row, column, resize_to=None):
    img = MyLoadImage(img_path, resize_to=resize_to)
    image = img.get_image()
    label_var = Tkinter.Label(frame, image=image)
    label_var.image = image
    label_var.grid(row=row, column=column)


def write_text(frame, text, row=0, column=0):
    label_text = Tkinter.Label(frame, text=text,
                               font=("Helvetica", 16),
                               foreground="green")
    label_text.grid(row=row, column=column)


def undistort(frame):
    calib = np.load('./data/calib.npz')
    mtx, dist = calib['mtx'], calib['dist']
    h, w = frame.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, mtx, dist, None, newCameraMtx)
    return frame
