import cv2
import mediapipe as mp
import time

class personDetector():
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, refine_face_landmarks=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpPerson = mp.solutions.holistic
        self.person = self.mpPerson.Holistic(self.mode, self.complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.refine_face_landmarks, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPerson(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.person.process(imgRGB)
        
        if self.results.left_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpPerson.HAND_CONNECTIONS)
        if self.results.right_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpPerson.HAND_CONNECTIONS)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPerson.POSE_CONNECTIONS)
        if self.results.face_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpPerson.FACEMESH_CONTOURS)
        
        return img

    def findPosition(self, img, draw=True):

        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

        if self.results.left_hand_landmarks:
            for id, lm in enumerate(self.results.left_hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        if self.results.right_hand_landmarks:
            for id, lm in enumerate(self.results.right_hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        if self.results.face_landmarks:
            for id, lm in enumerate(self.results.face_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
                
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = personDetector()

    while True:
        success, img = cap.read()
        img = detector.findPerson(img)
        lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()