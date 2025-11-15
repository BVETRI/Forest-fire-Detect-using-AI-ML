# camera.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
from PIL import Image
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        self.video = cv2.VideoCapture(0)
        self.k=1

        self.lower_bound = np.array([11,33,111])

        self.upper_bound = np.array([90,255,255])

        #cap = self.video
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('video.mp4')

        # Check if camera opened successfully
        #if (cap.isOpened() == False): 
        #  print("Unable to read camera feed")

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        #frame_width = int(cap.get(3))
        #frame_height = int(cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #self.out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


        
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, frame = self.video.read()
        cv2.imwrite("static/fire.jpg", frame)
        
        
        frame = cv2.resize(frame,(1280,720))
        frame = cv2.flip(frame,1)
        frame_smooth = cv2.GaussianBlur(frame,(7,7),0)
        mask = np.zeros_like(frame)
        mask[0:720, 0:1280] = [255,255,255]
        img_roi = cv2.bitwise_and(frame_smooth, mask)
        frame_hsv = cv2.cvtColor(img_roi,cv2.COLOR_BGR2HSV)
        image_binary = cv2.inRange(frame_hsv, self.lower_bound, self.upper_bound)

        check_if_fire_detected = cv2.countNonZero(image_binary)
        print(check_if_fire_detected)

        ff=open("value.txt","r")
        value=ff.read()
        ff.close()
        val=int(value)
        if int(check_if_fire_detected) >= val :
            ff=open("detect.txt","w")
            ff.write("detect")
            ff.close()
            cv2.putText(frame,"Fire Detected !",(300,60),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),2)

        else:
            ff=open("detect.txt","w")
            ff.write("")
            ff.close()
            

            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
