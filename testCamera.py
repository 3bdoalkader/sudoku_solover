import cv2 as cv
import sudokuSolver

from recognize import Recognize

model = Recognize()
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    sudoko = sudokuSolver.recognize_sudoku(frame,model)
    # sudoko = cv.resize(sudoko, (1500, 650))
    # _, sudoko = cv.threshold(sudoko, 200, 255, cv.THRESH_BINARY)

    cv.imshow('sudoko', sudoko)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
