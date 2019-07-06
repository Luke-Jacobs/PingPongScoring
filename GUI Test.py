import cv2
import numpy as np
# import PILasOPENCV as Image
from PIL import Image, ImageDraw, ImageFont
# import PILasOPENCV as ImageDraw
# import PILasOPENCV as ImageFont

LEFT = 0
RIGHT = 1

IMG_DISPLAY = Image.open('templates/display.png').convert('RGBA')
IMG_UNDERSCORE = Image.open('templates/white_underscore.jpg')
IMG_QUESTION_MARK = Image.open('templates/question_mark_full.png').convert('RGBA')

def getDisplay(score: list, serving: int):
    # QUESTION_MARK = (882, 49)
    LEFT_UNDERSCORE = (200, 313)
    RIGHT_UNDERSCORE = (1187, 313)
    SINGLE_DIGIT_RIGHT = (1365, 500)
    SINGLE_DIGIT_LEFT = (380, 500)
    DOUBLE_DIGIT_RIGHT = (1225, 500)
    DOUBLE_DIGIT_LEFT = (240, 500)

    display = IMG_DISPLAY.copy()

    if serving == LEFT:
        display.paste(IMG_UNDERSCORE, LEFT_UNDERSCORE)
    elif serving == RIGHT:
        display.paste(IMG_UNDERSCORE, RIGHT_UNDERSCORE)
    else:
        # Unknown side - ask the user
        display = Image.alpha_composite(display, IMG_QUESTION_MARK)

    font = ImageFont.truetype("font/Roboto-Regular.ttf", 150)
    draw = ImageDraw.Draw(display)

    if score[0] > 9:
        draw.text(DOUBLE_DIGIT_LEFT, str(score[0]), font=font, fill=(255, 255, 255))
    else:
        draw.text(SINGLE_DIGIT_LEFT, str(score[0]), font=font, fill=(255, 255, 255))

    if score[1] > 9:
        draw.text(DOUBLE_DIGIT_RIGHT, str(score[1]), font=font, fill=(255, 255, 255))
    else:
        draw.text(SINGLE_DIGIT_RIGHT, str(score[1]), font=font, fill=(255, 255, 255))

    cvImg = cv2.cvtColor(np.asarray(display), cv2.COLOR_RGB2BGR)
    return cvImg


if __name__ == '__main__':

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    display = getDisplay([5, 6], LEFT)

    while True:
        cv2.imshow("window", display)

        if cv2.waitKey(20) == ord('q'):
            break
