from dt_apriltags import Detector
import numpy
import os
import cv2

imagepath = './april_test_kitchen.jpg'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cameraMatrix = numpy.array([336.7755634193813, 0.0, 333.3575643300718, 0.0, 336.02729840829176, 212.77376312080065, 0.0, 0.0, 1.0]).reshape((3,3))
camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


tags = at_detector.detect(image, True, camera_params, 0.065)
print(tags)
