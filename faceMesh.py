import cv2
import pyautogui
import mediapipe as mp
import time
import autopy
import math
import numpy as np
from utilities import findCentreIris, landmarkDetect, colorBackgroundText

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
FONTS = cv2.FONT_HERSHEY_TRIPLEX



# Euclidean distance
def euclideanDistance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

#Blink Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    #cv2.line(img, rh_right, rh_left, (0,255,255), 2)
    #cv2.line(img, rv_top, rv_bottom, (0,0,255), 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)

    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    # reRatio = rhDistance/rvDistance
    # leRatio = lhDistance/lvDistance
    #
    # ratio = (reRatio+leRatio)/2
    ratio = (lvDistance+rvDistance)/2
    print(rvDistance)
    return ratio, rvDistance, lvDistance



wCam, hCam = 640, 480

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=0.2, circle_radius=0.2)


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = autopy.screen.size()
def main():
    pTime = 0
    close_eye_counter = 0
    total_blinks = 0

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.7) as face_mesh:
        while cap.isOpened():
            success, img = cap.read()

            # Improve performance  --> pass by reference the image as not writeable
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cTime = time.perf_counter()

            results = face_mesh.process(img)

            # Draw face mesh on image
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                norm_cord, rIris, lIris = landmarkDetect(img, results, draw=False)
                ratio, reRat, leRat = blinkRatio(img, norm_cord, RIGHT_EYE, LEFT_EYE)
                #print(norm_cord[RIGHT_EYE][0])

                # Find centre of iris
                img, centerRight, centerLeft, rightRad, leftRad = findCentreIris(img, rIris, lIris, (0, 255, 255))

                # Calculate circle for iris - Draw circle

                iLeft = cv2.circle(img, centerLeft, int(leftRad), (255,0,255), 1, cv2.LINE_AA)
                iRight = cv2.circle(img, centerRight, int(rightRad), (255, 0, 255), 1, cv2.LINE_AA)

                colorBackgroundText(img, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, (147,50,255),
                                    (0, 255, 255))

                if ratio > 5.5:
                    close_eye_counter += 1
                    # colorBackgroundText(img, f'Blink', FONTS, 1.7, (int(wScr / 2), 100), 2,
                    #                           (0, 255, 255), pad_x=6, pad_y=6, )

                else:
                    if close_eye_counter > 3:
                        total_blinks += 1
                        close_eye_counter = 0
                # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                colorBackgroundText(img, f'Total Blinks: {total_blinks}', FONTS, 0.7, (30, 150), 2, (147,50,255),
                                    (0, 255, 255))

            # Convert coords - Smooth values
            xSmooth = np.interp(centerRight[0], (0, wCam), (0, wScr))
            ySmooth = np.interp(centerRight[1], (0, hCam), (0, hScr))


            # Move mouse with right eye
            #autopy.mouse.move(wScr - xSmooth, ySmooth)
            pyautogui.moveTo(wScr - xSmooth, ySmooth)
            # #autopy.mouse.smooth_move(wScr - xSmooth, ySmooth)
            #
            # # If left eye closed left click - right eye right click
            if leRat < 3.5 and reRat > 5.5:
            #     #autopy.mouse.click()
                pyautogui.click()
                time.sleep(.1)
            if reRat < 3.5 and leRat > 5.5:
            #     #autopy.mouse.click()
                 pyautogui.click(button='right')
            #     time.sleep(.1)
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {str(int(fps))}', (30, 50), FONTS, 0.7, (147,50,255))
            cv2.imshow("Iris Detection", img)
            key = cv2.waitKey(1)
            if key == ord('q') or ratio > 23:
                    break

if __name__ == '__main__':
    main()