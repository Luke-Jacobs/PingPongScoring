from typing import Union, List, Optional
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import imutils
from imutils import video

# Shared constants needed for Game and Ball

class Convert:

    @staticmethod
    def blenderToCV2(*argv):
        conversionFactors = (179, 255, 255)
        return tuple([int(argv[i] * conversionFactors[i]) for i in range(len(argv))])

    @staticmethod
    def paintNetToCV2(*argv):
        Hue = int(argv[0] * (179/360.0))
        Sat = int(argv[1] * (255/100.0))
        Val = int(argv[2] * (255/100.0))
        return Hue, Sat, Val

    @staticmethod
    def CV2ToPaintNet(*argv):
        Hue = int(argv[0] * (360.0 / 179))
        Sat = int(argv[1] * (100.0 / 255))
        Val = int(argv[2] * (100.0 / 255))
        return Hue, Sat, Val

    @staticmethod
    def GIMPtoBlender(*argv):
        conversionFactors = (1/360, 1/100, 1/100)
        return tuple([float(argv[i] * conversionFactors[i]) for i in range(len(argv))])


class GameViewSource:

    def __init__(self, stream: Union[cv2.VideoCapture, video.webcamvideostream.WebcamVideoStream], isVideo: bool, res: tuple,
                 netX: Optional[int]=None):
        self.stream = stream
        self.isVideo = isVideo
        self.res = res
        self.fps = self._getProp(cv2.CAP_PROP_FPS)
        self.netX = netX

    def setNetPos(self, netX: int):
        self.netX = netX

    def _getProp(self, prop):
        if self.isVideo:
            return self.stream.get(prop)
        else:
            return self.stream.stream.get(prop)

    def read(self) -> np.ndarray:
        if self.isVideo:
            return self.stream.read()[1]
        else:
            return self.stream.read()


CAP_RESOLUTION = (640, 480)
CAP_FRAMERATE = 30

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

# Simple Utilities
def other(side):
    if side is None:
        raise RuntimeError('None given to other side function.')
    return LEFT if side == RIGHT else RIGHT
def display(var):
    return 'NONE' if var is None else ('LEFT' if var == LEFT else 'RIGHT')

# GUI Display
IMG_DISPLAY = Image.open('templates/display.png').convert('RGBA')
IMG_UNDERSCORE = Image.open('templates/white_underscore.jpg')
IMG_QUESTION_MARK = Image.open('templates/question_mark_full.png').convert('RGBA')

# Get OpenCV image for display
def getDisplay(score: list, serving: Union[int, None]) -> cv2.UMat:
    # QUESTION_MARK = (882, 49)
    LEFT_UNDERSCORE = (200, 313)
    RIGHT_UNDERSCORE = (1187, 313)
    SINGLE_DIGIT_RIGHT = (1365, 500)
    SINGLE_DIGIT_LEFT = (380, 500)
    DOUBLE_DIGIT_RIGHT = (1225, 500)
    DOUBLE_DIGIT_LEFT = (240, 500)

    display = IMG_DISPLAY.copy()

    if serving == LEFT:
        display.paste(IMG_UNDERSCORE, RIGHT_UNDERSCORE)
    elif serving == RIGHT:
        display.paste(IMG_UNDERSCORE, LEFT_UNDERSCORE)
    else:
        # Unknown side - ask the user
        display = Image.alpha_composite(display, IMG_QUESTION_MARK)

    font = ImageFont.truetype("font/Roboto-Regular.ttf", 160)  # 150
    draw = ImageDraw.Draw(display)

    if score[0] > 9:
        draw.text(DOUBLE_DIGIT_RIGHT, str(score[0]), font=font, fill=(255, 255, 255))
    else:
        draw.text(SINGLE_DIGIT_RIGHT, str(score[0]), font=font, fill=(255, 255, 255))

    if score[1] > 9:
        draw.text(DOUBLE_DIGIT_LEFT, str(score[1]), font=font, fill=(255, 255, 255))
    else:
        draw.text(SINGLE_DIGIT_LEFT, str(score[1]), font=font, fill=(255, 255, 255))

    cvImg = cv2.cvtColor(np.asarray(display), cv2.COLOR_RGB2BGR)
    return cvImg

# Wait for a user to hold out their paddle for a signal
def getSideSignal(view: GameViewSource, score: List[int], displayFull=True) -> Optional[int]:
    # Color constants
    PADDLE_LOWER_1 = Convert.blenderToCV2(.00, .49, .70)
    PADDLE_HIGHER_1 = Convert.blenderToCV2(.04, .62, 1.0)

    PADDLE_LOWER_2 = Convert.blenderToCV2(.97, .49, .70)
    PADDLE_HIGHER_2 = Convert.blenderToCV2(1.0, .62, 1.0)

    # Look for signal until found
    screenImg = getDisplay(score, None)
    frameN = 0
    while True:
        frameN += 1
        frame = view.read()

        if frame is None:
            print('End of stream.')
            exit(-1)

        # Crop out the paddles on the table
        height, width = frame.shape[:2]
        frame = frame[:3*height//5]

        # Color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, PADDLE_LOWER_1, PADDLE_HIGHER_1)
        mask2 = cv2.inRange(hsv, PADDLE_LOWER_2, PADDLE_HIGHER_2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        redAreaImg = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        redAreaImg = cv2.morphologyEx(redAreaImg, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(redAreaImg.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 12:
                side = LEFT if x < view.netX else RIGHT
                print('\tDetected %s paddle signal after %d frames' % (display(side), frameN))
                return side  # Return side of table that is serving

        if displayFull:
            cv2.imshow('window', screenImg)
            # cv2.imshow('window', frame)
            if cv2.waitKey(1) == ord('q'):
                return None


# These must be 0 and 1 because they correspond to indices
HORIZONTAL = 0
VERTICAL = 1

SCALING_FACTOR = (CAP_RESOLUTION[0]/640.0)
NET_HIT_BUFFER = int(35 * SCALING_FACTOR)  # Buffer to confirm the ball has fully crossed the net barrier
NET_VIEW_BUFFER = int(110 * SCALING_FACTOR)  # Buffer to crop out moving players on the other side of the table
TABLE_END_BUFFER = int(200 * SCALING_FACTOR)  # Buffer that places a confidence interval on bounces

