import cv2
import sys
import os
import math

def main():
    # Open video that you want to extract frames from
    cap = cv2.VideoCapture('Videos/' + sys.argv[1])
    i = 0
    
    videoName = sys.argv[1].split('.')[0]

    os.mkdir('Frames/' + sys.argv[1])
    pathOut = 'Frames/' + sys.argv[1] + '/' + videoName

    # Get framerate every four seconds
    frameRate = math.floor(cap.get(5))
    
    while(cap.isOpened()):
        # Current frame number
        frameId = cap.get(1)

        ret, frame = cap.read()
        
        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break
        
        # Save Frame every second into disk using imwrite method
        #if (frameId % frameRate == 0):
        cv2.imwrite(pathOut + str(i) + '.jpg', frame)
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()