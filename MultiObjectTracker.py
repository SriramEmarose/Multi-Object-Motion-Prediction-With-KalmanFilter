#--------------------------------------------------------------------
# Implements multiple objects motion prediction using Kalman Filter
#
# Author: Sriram Emarose [sriram.emarose@gmail.com]
#
#
#
#--------------------------------------------------------------------

import cv2 as cv
import numpy as np
import sys

MAX_OBJECTS_TO_TRACK = 10

# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted



#Performs required image processing to get ball coordinated in the video
class ProcessImage:

    def DetectObject(self):

        vid = cv.VideoCapture('balls.mp4')

        if(vid.isOpened() == False):
            print('Cannot open input video')
            return

        width = int(vid.get(3))
        height = int(vid.get(4))

        # Create Kalman Filter Object
        kfObjs = []
        predictedCoords = []
        for i in range(MAX_OBJECTS_TO_TRACK):
            kfObjs.append(KalmanFilter())
            predictedCoords.append(np.zeros((2, 1), np.float32))

        while(vid.isOpened()):
            rc, frame = vid.read()

            if(rc == True):
                coords = self.DetectBall(frame)

                for i in range(len(coords)):
                    if(i > MAX_OBJECTS_TO_TRACK):
                        break

                    #print (' circle ',i, ' ', coords[i][0], ' ', coords[i][1])
                    predictedCoords[i] = kfObjs[i].Estimate(coords[i][0], coords[i][1])
                    frame = self.DrawPredictions(frame, coords[i][0], coords[i][1], predictedCoords[i])

                cv.imshow('Input', frame)

                if (cv.waitKey(300) & 0xFF == ord('q')):
                    break

            else:
                break

        vid.release()
        cv.destroyAllWindows()

    # Segment the green ball in a given frame
    def DetectBall(self, frame):

        frameGrey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        frameGrey = cv.blur(frameGrey, (3, 3))

        circles = cv.HoughCircles(frameGrey, cv.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 1, maxRadius = 40)
        coords = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                coords.append((x,y))
            return coords

        return coords

    def DrawPredictions(self, frame, ballX, ballY, predictedCoords):
        # Draw Actual coords from segmentation
        cv.circle(frame, (int(ballX), int(ballY)), 20, [0, 0, 255], 2, 8)
        cv.line(frame, (int(ballX), int(ballY + 20)), (int(ballX + 50), int(ballY + 20)), [100, 100, 255], 2, 8)
        cv.putText(frame, "Actual", (int(ballX + 50), int(ballY + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

        # Draw Kalman Filter Predicted output
        cv.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
        cv.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
        cv.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

        return frame



#Main Function
def main():

    processImg = ProcessImage()
    processImg.DetectObject()


if __name__ == "__main__":
    main()

print('Program Completed!')