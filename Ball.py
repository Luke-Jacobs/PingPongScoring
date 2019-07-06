from collections import deque
import cv2
import numpy as np
from Setup import display, CAP_RESOLUTION, NET_HIT_BUFFER, NET_VIEW_BUFFER
from typing import Union, Tuple
import math
import imutils


LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

# These must be 0 and 1 because they correspond to indices
HORIZONTAL = 0
VERTICAL = 1

# Convert HSV values between formats
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


# Low level image processing / ball tracking
class Ball:

    YELLOW_LOWER = Convert.blenderToCV2(.09, .17, .24)
    YELLOW_HIGHER = Convert.blenderToCV2(.30, .75, 1.0)

    STATE_PRE_SERVE = 0
    STATE_ACTIVE_SERVE = 1
    STATE_FREE_BALL = 2
    STATE_EXPECTING_RETURN = 3
    STATE_TO_NAME = {
        STATE_FREE_BALL: "FB",
        STATE_ACTIVE_SERVE: "AS",
        STATE_PRE_SERVE: "PS",
        STATE_EXPECTING_RETURN: "ER"
    }

    N_POINTS = 5

    def __init__(self, netX, servingSide=None):
        # Constants
        self.netX = netX
        # Simple data
        self.pos = None
        self.lastPos = None
        self.lastDisp = None
        self.netSide = servingSide
        self.points = deque(maxlen=self.N_POINTS)  # Stores all the data we have on the ball - even if None
        self.motionPoints = deque(maxlen=self.N_POINTS)  # Stores only the motions we know about the ball
        # Processed data from most recent frame
        self.bounceSide = None  # Either None (no bounce), Left, or Right side
        self.timeSinceBounce = 0
        self.hitDirection = None  # Either None (no hit), or to the Left or Right side
        self.ballCrossedTo = None  # Either None (no cross), or Left or Right side
        self.currentDir = None  # Current direction of the ball
        self.hasHitNet = False  # Has the ball bounced off the net
        # Processed stateful data
        self.framesOnSide = 0
        # Debug
        self.framesProcessed = 0

    def updatePosFromFrame(self, prevFrame, currentFrame, showProcessedFrame=True, showMaskFrame=True, output=False,
                           debugWrite=False) -> bool:
        self.framesProcessed += 1
        infoFrame = None

        # For drawing
        if showProcessedFrame:
            infoFrame = currentFrame.copy()
            cv2.putText(infoFrame, str(self.framesProcessed), (5,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            # cv2.putText(infoFrame, str(SCORE), (100,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

        # Apply color threshold
        if prevFrame is None:
            return False

        # Get motion mask
        diffFrame = cv2.absdiff(currentFrame, prevFrame)
        grayDiffFrame = cv2.cvtColor(diffFrame, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(grayDiffFrame) == 0:  # If there is a duplicate frame or an unnecessary one
            return False
        ret, motionMask = cv2.threshold(grayDiffFrame, 15, 255, cv2.THRESH_BINARY)

        # Get color mask
        hsv = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2HSV)
        colorMask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_HIGHER)
        if debugWrite:
            cv2.imwrite('debug/frame_%d_color-mask.jpg' % self.framesProcessed, colorMask)

        # Combine masks and crop them to exclude moving players on other side
        maskTotal = cv2.bitwise_and(motionMask, colorMask)
        if debugWrite:
            cv2.imwrite('debug/frame_%d_total-mask.jpg' % self.framesProcessed, maskTotal)

        # Morphological operation
        kernel = np.ones((3,1), dtype=np.uint8)
        squareKernel = np.ones((5,5), dtype=np.uint8)
        maskTotal = cv2.morphologyEx(maskTotal, cv2.MORPH_DILATE, kernel)
        maskTotal = cv2.morphologyEx(maskTotal, cv2.MORPH_OPEN, squareKernel)
        if debugWrite:
            cv2.imwrite('debug/frame_%d_morph-mask.jpg' % self.framesProcessed, maskTotal)

        # Crop frame to exclude moving players on the other side of the table
        # This forces the ball to only cross sides through the central view buffer
        if self.netSide == LEFT:
            maskTotal[:, self.netX + NET_VIEW_BUFFER:] = 0
        elif self.netSide == RIGHT:
            maskTotal[:, :self.netX - NET_VIEW_BUFFER] = 0

        # If we know the motion of the ball, we can assume where it will be headed
        if self.lastDisp is not None and self.pos is not None:
            VERTICAL_BUFFER = 2 * abs(self.lastDisp[1]) + 50
            HORIZONTAL_BUFFER = 2 * abs(self.lastDisp[0]) + 40
            SHIFT_X = ((CAP_RESOLUTION[0] - 2 * self.lastPos[0]) / (2 * CAP_RESOLUTION[0])) * 2.5
            if output: print('Shift X: %f' % SHIFT_X)
            POINT = (int((self.lastPos[0]+self.lastDisp[0])+SHIFT_X*HORIZONTAL_BUFFER), self.lastPos[1]+self.lastDisp[1])
            top_left = (POINT[0]-HORIZONTAL_BUFFER if POINT[0]-HORIZONTAL_BUFFER > 0 else 0, POINT[1]-VERTICAL_BUFFER)
            bot_right = (POINT[0]+HORIZONTAL_BUFFER if POINT[0]+HORIZONTAL_BUFFER > 0 else 0, POINT[1]+VERTICAL_BUFFER)

            blank = np.zeros((CAP_RESOLUTION[1], CAP_RESOLUTION[0]), dtype=np.uint8)
            blank[top_left[1]:bot_right[1], top_left[0]:bot_right[0]] = \
                maskTotal[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
            maskTotal = blank
            if showProcessedFrame: cv2.rectangle(infoFrame, top_left, bot_right, (255, 0, 0), 2)

        # Refer to identification function
        center = self._identifyBallCenter(maskTotal, self.lastPos, output=output)

        # Process ball point
        self.points.append(center)
        self.pos = center
        if center is not None:
            if self.lastPos is not None:
                self.lastDisp = (center[0]-self.lastPos[0], center[1]-self.lastPos[1])
                self.currentDir = LEFT if self.lastDisp[0] < 0 else RIGHT
            self.lastPos = center
            self.motionPoints.append(center)

        # Draw pretty lines
        if showProcessedFrame:
            for i in range(1, len(self.motionPoints)):
                infoFrame = cv2.line(infoFrame, tuple(self.motionPoints[i - 1]), tuple(self.motionPoints[i]),
                                     (0, 0, 255), 2)

        if showProcessedFrame and showMaskFrame:
            maskTotal = cv2.cvtColor(maskTotal, cv2.COLOR_GRAY2BGR)
            bothImgs = np.hstack((infoFrame, maskTotal))
            cv2.imshow('Both frames', bothImgs)
        elif showProcessedFrame:
            # Process GUI
            cv2.imshow('Processed frame', infoFrame)
        elif showMaskFrame:
            cv2.imshow('Total mask frame', maskTotal)

        return True

    def updateProcessedData(self, output: bool=False) -> None:
        # Net Side
        if self.pos is None:
            newNetSide = self.netSide
            # print('\tNet Side: Assuming ball is out of view on the %s side' %
            #       ('left' if self.netSide == LEFT else 'right'))
        else:
            newNetSide = LEFT if self.pos[0] < self.netX else RIGHT  # FIXME maybe buffer
            # print('\tNet Side: %s' % ('left' if newNetSide == LEFT else 'right'))

        # Net change
        if newNetSide != self.netSide:
            if output: print('Ball net change')
            self.framesOnSide = 0
            self.ballCrossedTo = newNetSide
        else:
            self.framesOnSide += 1
            self.ballCrossedTo = None
        self.netSide = newNetSide

        # Detect recent bounce or paddle hit
        if len(self.motionPoints) > 2:
            last3points = list(self.motionPoints)[-3:]
            self.bounceSide = self._detectTableBounce(last3points, self.netX)
            self.timeSinceBounce = -1
            self.hitDirection = self._detectPaddleHit(last3points)
        self.timeSinceBounce += 1

        # Detect net hit
        if len(self.motionPoints) > 4:
            self.hasHitNet = self._hasHitNet(list(self.motionPoints)[-5:], self.netX)

        # Display data
        if output:
            print('---- Ball ----\nSide: %s (%d)\nBounce: %s\nHit: %s\nNet Hit: %s' %
              (display(newNetSide), self.framesOnSide,
               display(self.bounceSide), display(self.hitDirection), str(self.hasHitNet)))

    @staticmethod
    def _identifyBallCenter(mask, lastPos: tuple, output: bool=False) -> Union[Tuple[int, int], None]:
        """Process the contours of a mask and extract the most likely center of the ball."""

        # Find contours in combined mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Iterate through contours and return the center of the one that is the most likely candidate
        center = None
        cntsSortedByArea = sorted(cnts, key=cv2.contourArea, reverse=True)
        for contour in cntsSortedByArea:  # Run through the biggest contours
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            potentialPos = (int(x), int(y))
            # infoFrame = cv2.circle(infoFrame, potentialPos, int(radius), (0, 255, 255), 2)
            if radius > 1:
                if radius > 20.0:
                    if output: print('Contour is definitely big enough to be the ball (radius = %f)' % radius)
                    center = (int(x), int(y))
                    break
                if lastPos is not None:
                    deltaDistance = Ball._distance(Ball._displacement(potentialPos, lastPos))
                    if output: print('Change in difference: %f' % deltaDistance)
                    if deltaDistance < 120.0:
                        if output: print('Within acceptable boundaries')
                        center = potentialPos
                        break
                    else:
                        if output: print('Too far of a jump')
                else:
                    center = potentialPos
                    break

        return center

    @staticmethod
    def _getDisplacements(motionPts: list) -> list:
        # Compute displacements between motion points
        displacementPts = []
        for i in range(len(motionPts) - 1):
            dispPt = (motionPts[i + 1][0] - motionPts[i][0], motionPts[i + 1][1] - motionPts[i][1])
            displacementPts.append(dispPt)
        return displacementPts

    @staticmethod
    def _hasChangedDirection(motionPts: list) -> tuple:
        """Looks for changes in direction across many motion points and for both x and y axes."""
        dispPts = Ball._getDisplacements(motionPts)
        xDir = yDir = None
        xChange = yChange = False
        for dispPt in dispPts:
            # Compute differences
            xDirNow = RIGHT if dispPt[0] > 0 else LEFT
            yDirNow = DOWN if dispPt[1] > 0 else UP
            # Look for x changes
            if xDir is None:
                xDir = xDirNow
            elif xDirNow != xDir:
                xChange = True
            # Look for y changes
            if yDir is None:
                yDir = yDirNow
            elif yDirNow != yDir:
                yChange = True
        return xChange, yChange

    @staticmethod
    def _hasChangedAxisDirectionAt(motionPts: list, axis: int) -> tuple or None:
        """
        Looks for changes in direction across many motion points and for a certain axes.
        Returns None if there is no change, else returns the point where the ball changed direction on that axis
        """
        dispPts = Ball._getDisplacements(motionPts)
        lastDir = None
        dirChange = False
        for i, dispPt in enumerate(dispPts):
            # Compute difference - 0 and 1 are arbitrary
            currentDir = 0 if dispPt[axis] > 0 else 1
            # Look for changes
            if lastDir is None:
                lastDir = currentDir
            elif currentDir != lastDir:
                return motionPts[i]
        return None

    @staticmethod
    def _detectTableBounce(motionPts: list, netX) -> int or None:
        if len(motionPts) != 3:
            raise RuntimeError('To detect a bounce, you need 3 motion points.')

        # If the ball changes horizontal velocity, it has not bounced off of the table
        if Ball._hasChangedDirection(motionPts)[0]:
            return None

        dispPts = Ball._getDisplacements(motionPts)
        lastYangle = None
        for disp in dispPts:
            if disp[0] == 0:
                disp = (1, disp[1])

            if lastYangle is None:
                lastYangle = disp[1] / abs(disp[0])
            # If downward trajectory is less than in the previous displacement,
            # then the ball must have bounced
            if 0 < lastYangle > disp[1] / abs(disp[0]):
                # Bounce detected, now we need to compute which side it was on
                if motionPts[1][0] > netX:
                    return RIGHT
                else:
                    return LEFT

        return None

    @staticmethod
    def _detectPaddleHit(motionPts: list) -> int or None:
        if len(motionPts) != 3:
            raise RuntimeError('Need 3 motion points to detect a hit.')

        # Check for horizontal direction change
        if Ball._hasChangedDirection(motionPts)[0]:
            # Figure out what direction the hit was towards
            lastDisplacement = motionPts[-1][0] - motionPts[-2][0]
            if lastDisplacement > 0:
                return RIGHT
            else:
                return LEFT
        else:
            return None

    @staticmethod
    def _hasCrossedDistance(motionPts: list, xLeft: int, xRight: int):
        # This can be very much optimized
        if Ball._hasCrossedLine(motionPts, xLeft) and Ball._hasCrossedLine(motionPts, xRight):
            return True
        else:
            return False

    @staticmethod
    def _hasCrossedLine(motionPts: list, line: int):
        before = after = False
        for pt in motionPts:
            if pt[0] > line:
                after = True
            if pt[0] < line:
                before = True
        if before and after:
            return True
        else:
            return False

    @staticmethod
    def _hasEnteredNotCrossed(motionPts: list, xLeft: int, xRight: int):
        # This can be very much optimized - cross one line but not the other
        if Ball._hasCrossedLine(motionPts, xLeft) ^ Ball._hasCrossedLine(motionPts, xRight):
            return True
        else:
            return False

    @staticmethod
    def _hasHitNet(motionPts: list, netX: int) -> bool:
        """Requires about 5 motion points for an accurate response."""

        # Need to change horizontal direction

        # Changing horizontal direction near the net is a requirement for hitting the net
        leftSide = netX - NET_VIEW_BUFFER
        rightSide = netX + NET_VIEW_BUFFER
        point = Ball._hasChangedAxisDirectionAt(motionPts, HORIZONTAL)
        # If no direction change at all
        if point is None:
            return False
        # If the ball is not near the net when it changes direction
        if not (leftSide < point[0] < rightSide):
            return False

        # Crossing the net hit boundary disqualifies the motion as a net hit
        if Ball._hasCrossedDistance(motionPts, netX - NET_HIT_BUFFER, netX + NET_HIT_BUFFER):
            print('HN: Has crossed distance')
            return False

        # Must enter the boundary
        if not Ball._hasEnteredNotCrossed(motionPts, netX - NET_HIT_BUFFER, netX + NET_HIT_BUFFER):
            # print('HN: Has not entered boundary')
            return False

        # A point must be below a certain height - must be under the net's height
        NET_TOP = CAP_RESOLUTION[1] - (CAP_RESOLUTION[1] // 3)
        for pt in motionPts:
            if pt[1] > NET_TOP:
                return True
        # print('HN: Too tall')
        return False

    @staticmethod
    def _distance(displacementPt) -> float:
        return math.sqrt(displacementPt[0]**2 + displacementPt[1]**2)

    @staticmethod
    def _displacement(pt1, pt2) -> tuple:
        return pt1[0]-pt2[0], pt1[1]-pt2[1]

