import cv2
import attendance_proj as ap


cap = cv2.VideoCapture(0)
encode_img, class_name = ap.get_data('attendance_proj/classinfo')
encodelist = ap.findEncoding(encode_img)

while True:
    _, new_img = cap.read()
    ap.verifynew(new_img, encodelist, class_name)


