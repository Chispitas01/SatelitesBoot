from ultralytics import YOLO
import cv2

model=YOLO("best.pt")
cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    result=model.predict(frame, imgsz=640, conf=0.98)

    anot=result[0].plot()

    cv2.imshow("deteccion y segmentacion", anot)

    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()