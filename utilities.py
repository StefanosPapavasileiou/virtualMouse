import cv2
import numpy as np

# Double click -> Both eyes blink
# Right click -> Right eye blink

'''MediaPipe returns normalized values
    we need to convert in pixels'''

def landmarkDetect(img, results, draw=False):
    '''Return list of tuples (x, y) for each landmark'''
    img_h, img_w = img.shape[:2]

    #Extract x,y coordinates from mesh landmarks
    mesh_coords = np.array([(int(point.x * img_w),  int(point.y * img_h)) for point in results.multi_face_landmarks[0].landmark])

    #Extract tuple (x,y) with 4 landmarks for each Iris
    rightIris = mesh_coords[469:473]
    leftIris = mesh_coords[474:478]

    if draw:
        # Right , Left iris draw
        [cv2.circle(img, p, 2, (255,255,0), -1) for p in mesh_coords[469:473]]
        [cv2.circle(img, q, 2, (255,255,0), -1) for q in mesh_coords[474:478]]

    return mesh_coords, rightIris, leftIris

def fillPolyCustom(img, pts, color, opacity):
    '''

    :param img: input image
    :param pts: list tuples as (int, int)
    :param color: (int, int , int)
    :param opacity: how much the color of poly would finally see
    :return: image drawn with Poly on top
    '''

    # First need to convert to np array with dtype as int32
    npList = np.array(pts, dtype=np.int32)
    over = img.copy()
    cv2.fillPoly(over, [npList], color)
    # Combine 2 img as 1
    newImg = cv2.addWeighted(over, opacity, img, 1 - opacity, 0)
    return newImg


def findCentreIris(img, lIris, rIris, color):
    '''

    :param img: input image
    :param left: left iris list of (x, y) tuples
    :param right: right iris list of (x, y) tuples
    :param color: (int, int, int)
    :return: image drawn with centre of iris
    '''

    #rCx, rCy = (rIris[0][0] + rIris[1][0] + rIris[2][0] + rIris[3][0]) // 4, \
    #            (rIris[0][1] + rIris[1][1] + rIris[2][1] + rIris[3][1]) // 4

    #lCx, lCy = (lIris[0][0] + lIris[1][0] + lIris[2][0] + lIris[3][0]) // 4, \
    #            (lIris[0][1] + lIris[1][1] + lIris[2][1] + lIris[3][1]) // 4


    (rCx, rCy), r_rad = cv2.minEnclosingCircle(rIris)
    center_right = np.array([rCx, rCy], dtype=np.int32)

    (lCx, lCy), l_rad = cv2.minEnclosingCircle(lIris)
    center_left = np.array([lCx, lCy], dtype=np.int32)

    newImg = cv2.circle(img, center_right, radius=1, color=color, thickness=1)
    newImg = cv2.circle(newImg, center_left, radius=1, color=color, thickness=1)


    return newImg, center_right, center_left, r_rad, l_rad

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param bgColor: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background
    @return: img(mat) with draw with background
    """
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv2.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle
    cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img