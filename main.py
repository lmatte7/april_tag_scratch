from dt_apriltags import Detector
import numpy
import os
import cv2


at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    tags = at_detector.detect(gray, False)
    if len(tags) > 0:
        tag = tags[0]
        top_left_x = int(tag.corners[1][0])
        top_left_y = int(tag.corners[1][1])
        bottom_left_x = int(tag.corners[3][0])
        bottom_left_y = int(tag.corners[3][1])
        cv2.rectangle(gray,(top_left_x, top_left_y),(bottom_left_x, bottom_left_y),(0,255,0),3)
        print(tags[0])
        print("\n")

    # Display the resulting frame
    cv2.imshow('frame', gray)      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 