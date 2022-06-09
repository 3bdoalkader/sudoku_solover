import cv2
import numpy as np


class Recognize:
    def __init__(self):
        #######   training part    ###############
        samples = np.loadtxt('generalsamples.data', np.float32)
        responses = np.loadtxt('generalresponses.data', np.float32)
        responses = responses.reshape((responses.size, 1))

        self.model = cv2.ml.KNearest_create()
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    ############################# testing part  #########################
    def recognizing(self, image):
        image = cv2.resize(image, (50, 50))
        thresh = cv2.adaptiveThreshold(image, 255, 1, 1, 11, 2)
        # cv2.imshow("th",thresh)
        # cv2.waitKey(0)

        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > 28:
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    roismall = roismall.reshape((1, 100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = self.model.findNearest(roismall, k=1)
                    return str(int((results[0][0])))
        return 0
