import cv2
import mediapipe as mp
import time
import numpy as np

class handDetector():
    def __init__(self, mode: bool = False,
                 maxhands:int = 2,
                 detectionCon:float = 0.5,
                 trackCon:float = 0.5):
        """
        Initializes a new instance of the `handDetector` class.

        Args:
            mode (bool, optional): Specifies the mode of the hand detector. Defaults to False.
            maxhands (int, optional): Specifies the maximum number of hands to detect. Defaults to 2.
            detectionCon (float, optional): Specifies the confidence threshold for hand detection. Defaults to 0.5.
            trackCon (float, optional): Specifies the confidence threshold for hand tracking. Defaults to 0.5.
        """
        self.mode = mode
        self.maxHands = maxhands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode,
                                        max_num_hands = self.maxHands,
                                        min_detection_confidence = self.detectionCon,
                                        min_tracking_confidence = self.trackCon
                                        )
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw = True):
        """
        Finds and draws the hands in the image.

        Args:
            img (numpy.ndarray): The image to find the hands in.
            draw (bool, optional): Whether to draw the hands on the image. Defaults to True.

        Returns:
            numpy.ndarray: The image with the hands drawn on it.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,
                                               handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    
        return img
    
    def findPosition(self, img:np.ndarray, handNo = 0, draw = True):
        """
        Finds the positions of the hands in the image.

        Args:
            img (numpy.ndarray): The image to find the positions of the hands in.
            handNo (int, optional): The number of the hand to find the positions of. Defaults to 0.
            draw (bool, optional): Whether to draw the positions of the hands on the image. Defaults to True.

        Returns:
            list: A list of tuples (id, x, y) containing the positions of the hands.
        """
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw and id in [8]:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 355), cv2.FILLED)
        return lmList
                
def main():
    # Example code
    
    # from handTrackingModule import handDetector
    # import cv2
    # import time
    # import mediapipe as mp
    
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img=img, draw=True) 
        
        if len(lmlist) != 0:
            print(lmlist[4])
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)),
                    (10, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()