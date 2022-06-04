import sys
import torch
import numpy as np
import cv2
import imageio
import time
from ball_tracker import BallDetection
from player_tracker import PlayerDetection
import os #OpenMP bug
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sort import *


def main():
    """
        This function runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
    """
    ballDetector = BallDetection(capture_index='Videos/' + sys.argv[1], model_name='best.pt')
    
    playerDetector = PlayerDetection(capture_index='Videos/' + sys.argv[1], model_name=None)

    mot_tracker = Sort()

    cap = playerDetector.get_video_capture()
    assert cap.isOpened()
    #image_lst = []
      
    while True:
        
        ret, frame = cap.read()
        if not ret:
            return True
        
        frame = cv2.resize(frame, (608,	352))
        
        start_time = time.time()

        playerFrame = playerDetector.plot_boxes(playerDetector.score_frame(frame, mot_tracker), frame)
        newFrame = ballDetector.plot_boxes(ballDetector.score_frame(frame), playerFrame)
        
        end_time = time.time()
        fps = 1/np.round(end_time - start_time, 2)
        #print(f"Frames Per Second : {fps}")
            
        cv2.putText(newFrame, f'FPS: {int(fps)}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,255), 2)
        #image_lst.append(cv2.cvtColor(newFrame, cv2.COLOR_BGR2RGB))
        cv2.imshow('YOLOv5 Detection', newFrame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    #imageio.mimsave('video.gif', image_lst, fps=30)
    return True

if __name__ == "__main__":
    main()