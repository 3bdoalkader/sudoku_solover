import cv2 as cv
import sudokuSolver
from recognize import Recognize

model = Recognize()
pre_sudoku=None
soluation=None
image=cv.imread("sudoku_2.png")
image=sudokuSolver.recognize_sudoku(image,model)
# _, image = cv.threshold(image, 200, 255, cv.THRESH_BINARY)
cv.imshow("sudoku",image)

cv.waitKey(0)