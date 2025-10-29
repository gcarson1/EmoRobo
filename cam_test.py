import cv2

print("Checking available camera indexes...")
for i in range(0, 10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} works.")
        cap.release()
    else:
        print(f"Camera index {i} not available.")
