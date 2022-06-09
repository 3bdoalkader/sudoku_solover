import numpy as np
import cv2 as cv
import math
from recognize import Recognize

import sudoku
import copy


def recognize_sudoku(image,model):

    # Convert to a gray image, blur that gray image for easier detection
    # and apply adaptiveThreshold
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 225, 1, 1, 11, 2)

    # Find all contours
    _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Extract countour with biggest area, assuming the Sudoku board is the BIGGEST contour
    max_area = 0
    biggest_contour = None
    for c in contours:
        area = cv.contourArea(c)
        if area > max_area:
            max_area = area
            biggest_contour = c
    if biggest_contour is None:  # If no sudoku
        return image

    # Get 4 corners of the biggest contour
    corners = get_corners(biggest_contour, len(biggest_contour))
    if corners is None:  # If no sudoku
        return image

    bottom_right, top_left, bottom_left, top_right = sort_corner(corners)


    # After having found 4 corners A B C D, check if ABCD is approximately square
    A = top_left[0]
    B = top_right[0]
    C = bottom_right[0]
    D = bottom_left[0]


    # 1st condition: If all 4 angles are not approximately 90 degrees (with tolerance = epsAngle), stop
    AB = B - A  # 4 vectors AB AD BC DC
    AD = D - A
    BC = C - B
    DC = C - D
    eps_angle = 10
    if not (approx_90_degrees(angle_between(AB, AD), eps_angle) and approx_90_degrees(angle_between(AB, BC), eps_angle)
            and approx_90_degrees(angle_between(BC, DC), eps_angle) and approx_90_degrees(angle_between(AD, DC),eps_angle)):
        return image

    # 2nd condition: The Lengths of AB, AD, BC, DC have to be approximately equal
    # => Longest and shortest sides have to be approximately equal
    eps_scale = 1.3 # Longest cannot be longer than epsScale * shortest
    if (side_lengths_are_too_different(A, B, C, D, eps_scale)):
        return image

    pts1 = np.float32([[bottom_right[0][0], bottom_right[0][1]],
                       [bottom_left[0][0], bottom_left[0][1]],
                       [top_left[0][0], top_left[0][1]],
                       [top_right[0][0], top_right[0][1]]])
    pts2 = np.float32([[252, 252], [0, 252], [0, 0], [252, 0]])
    matrix= cv.getPerspectiveTransform(pts1, pts2)
    warp = cut_image(image,matrix)

    orginal_warp = np.copy(warp)

    cv.imshow("imagewarp", warp)
    # cv.imwrite("imagewarp.png",warp,)
    # cv.waitKey(0)

    sol=cut_cell(warp,model)
    result=invers_perspectiv(image,sol,matrix)

    return result
def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon

def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    AD = math.sqrt((A[0] - D[0]) ** 2 + (A[1] - D[1]) ** 2)
    BC = math.sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2)
    CD = math.sqrt((C[0] - D[0]) ** 2 + (C[1] - D[1]) ** 2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest


# Return the angle between 2 vectors in degrees
def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    dot_droduct = np.dot(unit_vector_1, unit_vector2)
    angle = np.arccos(dot_droduct)
    return angle * 57 # Convert to degree

def invers_perspectiv(image, orginal_warp,matrix):
    result_sudoku = cv.warpPerspective(orginal_warp,matrix, (image.shape[1], image.shape[0])
                                       , flags=cv.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, image)
    return result


def get_corners(contours, max_iter):
    coefficient = 0.5
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1

        epsilon = coefficient * cv.arcLength(contours, True)

        poly_approx = cv.approxPolyDP(contours, epsilon, True)
        hull = cv.convexHull(poly_approx)
        if len(hull) == 4:
            return hull
        else:
            if len(hull) > 4:
                coefficient += .01
            else:
                coefficient -= .01
    return None


def cut_cell(image,model):
    warp=np.copy(image)
    warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
    warp = cv.GaussianBlur(warp, (5, 5), 0)
    warp = cv.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = cv.bitwise_not(warp)
    _, warp = cv.threshold(warp, 150, 255, cv.THRESH_BINARY)



    SIZE = 9
    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)
    thikLine = 3
    for j in range(9):
        for i in range(9):
            pts1 = np.float32(
                [[(i * 28) + thikLine, (j * 28) + thikLine], [(i * 28) + thikLine, ((j + 1) * 28) - thikLine],
                 [((i + 1) * 28) - thikLine, ((j + 1) * 28) - 2], [((i + 1) * 28) - thikLine, (j * 28) + thikLine]])
            pts2 = np.float32([[0, 0], [0, 252], [252, 252], [252, 0]])
            m = cv.getPerspectiveTransform(pts1, pts2)
            dst = cv.warpPerspective(warp, m, (252, 252))
            # feature_image(dst)
            digit_pic_size = 28
            crop_image = cv.resize(dst, (digit_pic_size, digit_pic_size))

            if crop_image.sum() >= digit_pic_size ** 2 * 255 - digit_pic_size * 1 * 255:
                grid[j][i] = 0

            else:

                grid[j][i] = (int)(model.recognizing(image=crop_image))
            # cv.imshow("c",crop_image)
            # print(grid[j][i])
            # cv.waitKey(0)
    user_grid = copy.deepcopy(grid)
    sudoku.solve_sudoku(grid)
    return write_solution_on_image(image, grid, user_grid)
    # cv.imshow("sol", sol)
    # cv.waitKey(0)

def sort_corner(corners):
    value = corners[0][0][0] + corners[0][0][1]
    bottom_right = corners[0]
    for c in corners:
        if value < c[0][0] + c[0][1]:
            bottom_right = c
            value = c[0][0] + c[0][1]

    value = corners[0][0][0] + corners[0][0][1]
    top_left = corners[0]
    for c in corners:
        if value > c[0][0] + c[0][1]:
            top_left = c
            value = c[0][0] + c[0][1]

    value = corners[0][0][0] - corners[0][0][1]
    top_right = corners[0]
    for c in corners:
        if value < c[0][0] - c[0][1]:
            top_right = c
            value = c[0][0] - c[0][1]

    value = corners[0][0][0] - corners[0][0][1]
    bottom_left = corners[0]
    for c in corners:
        if value > c[0][0] - c[0][1]:
            bottom_left = c
            value = c[0][0] - c[0][1]

    # print("bottom_right", bottom_right)
    # print("top_left", top_left)
    # print("bottom_left", bottom_left)
    # print("top_right", top_right)
    return bottom_right, top_left, bottom_left, top_right


def cut_image(image, matrix):

    dst = cv.warpPerspective(image, matrix, (252, 252))
    return dst


def feature_image(image):
    img = cv.resize(image.copy(), (250, 250))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (3, 3))
    thresh = cv.adaptiveThreshold(blur, 225, 0, 1, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    sift = cv.SIFT.create()
    kp = sift.detect(closing, None)

    img2 = cv.drawKeypoints(img, kp, None, color=(240, 0, 255))
    # cv.imshow("img", img2)
    # print(np.array(cv.KeyPoint_convert(kp)))
    # cv.waitKey(0)


def write_solution_on_image(image, grid, user_grid):
    if np.count_nonzero(grid)!=81:
        return image
    # Write grid on image
    size = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(size):
        for j in range(size):
            color=(0, 100,255)
            if user_grid[i][j] != 0:  # If user fill this cell
                color = (100, 200, 0)
            #     continue  # Move on
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), base_line = cv.getTextSize(text, font, fontScale=0.8, thickness=3)
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)

            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width * j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height * (i + 1) - math.floor((height - text_height) / 2) + off_set_y
            image = cv.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y),
                               font, font_scale,color , thickness=1, lineType=cv.LINE_8)
            # cv.imshow("sukk", image)
    return image