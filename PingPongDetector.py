import cv2
from imutils.video import VideoStream
import argparse
import time
from queue import Queue
from typing import Optional, Union
from Setup import NET_HIT_BUFFER, NET_VIEW_BUFFER, TABLE_END_BUFFER, \
    display, getSideSignal, getDisplay, GameViewSource, other
from Ball import Ball
from Game import GameState
from copy import copy
import numpy as np


def setupArguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default=False, help="Video to input as source")
    ap.add_argument("-f", "--writeFrame", default=False, action='store_true', help="Write masked image to output.avi")
    ap.add_argument("-m", "--writeMasked", default=False, action='store_true')
    ap.add_argument("-s", "--speed", default=1.0, type=float, help="Adjust the speed of playback")
    return vars(ap.parse_args())


def userSetupScene(view: GameViewSource) -> int:
    WINDOW_NAME = 'Setup Scene'
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)

    # Fields we need to get from the user
    netXvar = Queue()

    # Read scene img from stream
    frame = view.read()
    height, width = frame.shape[:2]
    print("Detected image is %d high" % height)

    # End of video
    if frame is None:
        print("Stream ended.")
        exit(-1)

    cv2.namedWindow(WINDOW_NAME)

    # Callback function
    def place_line(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            netXvar.put(x)

    cv2.setMouseCallback(WINDOW_NAME, place_line)

    startEnd = netX = None
    while True:
        cv2.imshow(WINDOW_NAME, frame)
        if netXvar.qsize() > 0:
            netX = netXvar.get()

            # Show net boundary
            cv2.line(frame, (netX, height), (netX, 0), BLUE, 2)
            # Show net hit buffer
            cv2.line(frame, (netX + NET_HIT_BUFFER, height), (netX + NET_HIT_BUFFER, 0), WHITE, 2)
            cv2.line(frame, (netX - NET_HIT_BUFFER, height), (netX - NET_HIT_BUFFER, 0), WHITE, 2)
            # Show net view buffer
            cv2.line(frame, (netX + NET_VIEW_BUFFER, height), (netX + NET_VIEW_BUFFER, 0), RED, 2)
            cv2.line(frame, (netX - NET_VIEW_BUFFER, height), (netX - NET_VIEW_BUFFER, 0), RED, 2)
            # Show table end buffer
            cv2.line(frame, (netX + TABLE_END_BUFFER, height), (netX + TABLE_END_BUFFER, 0), BLACK, 2)
            cv2.line(frame, (netX - TABLE_END_BUFFER, height), (netX - TABLE_END_BUFFER, 0), BLACK, 2)
            # Show signal line
            cv2.line(frame, (0, 3*height//5), (width, 3*height//5), (255, 255, 0), 2)

            print("The X position of the net has been set to: %d" % netX)
            startEnd = time.time()
        if cv2.waitKey(33) == ord('q'):
            print("Quiting interactive setup window")
            exit(-1)
        if startEnd is not None:
            if startEnd + 2 < time.time():
                cv2.destroyWindow(WINDOW_NAME)
                return netX


class GameMonitor:

    def __init__(self, oldGameState: GameState):
        self.oldGameState = copy(oldGameState)
        self.currentGame = oldGameState
        self.timeSinceRoundChange = time.time()
        self.gameClockOn = False

    def printWithTime(self, *args):
        timeMsg = ('[%04d]' % int(time.time() - self.timeSinceRoundChange)) if self.gameClockOn else '[STOP]'
        print(timeMsg, *args)

    def printNewEvents(self):
        # If new score
        if self.oldGameState.score != self.currentGame.score:
            self.printWithTime('%s has scored! Score is now: %s' %
                               (('Left' if self.currentGame.score[0] != self.oldGameState.score[0] else 'Right'),
                                str(self.currentGame.score)))

        # If the game has changed states
        if self.oldGameState.state != self.currentGame.state:
            stateMsg = ", "
            if self.currentGame.state == GameState.STATE_AMBIGUOUS_BOUNCE:
                stateMsg += 'on the %s' % display(self.currentGame.ambiguousBounceSide)
            elif self.currentGame.state == GameState.STATE_FREE_BALL:
                stateMsg += 'from the %s' % display(self.currentGame.freeBallFrom)
            elif self.currentGame.state == GameState.STATE_EXPECTING_RESPONSE:
                stateMsg += 'expecting %s to respond' % display(self.currentGame.expectingResponseFrom)
            elif self.currentGame.state == GameState.STATE_PRE_SERVE:
                if self.oldGameState.state == GameState.STATE_AMBIGUOUS_BOUNCE:
                    stateMsg += 'ambiguity resolved, ' % display(self.currentGame.servingSide)
                stateMsg += '%s is serving' % display(self.currentGame.servingSide)
                if not self.currentGame.serveCrossedNet:
                    # print('Game clock off')
                    self.gameClockOn = False
            self.printWithTime('State is now %s%s' % (self.currentGame.STATE_TO_NAME[self.currentGame.state], stateMsg))
        # If the game is giving the server an extra try
        elif self.oldGameState.givenSecondTry != self.currentGame.givenSecondTry:
            self.printWithTime('Giving server second try')
        # Serve crossed net
        elif self.oldGameState.serveCrossedNet != self.currentGame.serveCrossedNet:
            if self.currentGame.serveCrossedNet:
                self.gameClockOn = True
                self.timeSinceRoundChange = time.time()
                self.printWithTime('Serve crossed net to the %s side, round has started' % display(other(self.currentGame.servingSide)))

        self.oldGameState = copy(self.currentGame)

    def getGameDisplay(self) -> Optional[np.ndarray]:
        if self.currentGame.currentDisplay is not None:
            return self.currentGame.currentDisplay
        return None


def scoreGame(view: GameViewSource, showDisplay: bool=False, showFullDisplay: bool=False, slowDown: int=1):
    """The main game function."""

    prevFrame = None

    if showFullDisplay:
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    servingSide = getSideSignal(view, [0, 0], showFullDisplay)
    if servingSide is None:
        print('Quiting...')
        exit(-1)

    ball = Ball(view.netX, servingSide=servingSide)
    game = GameState(view)
    game.begin(view.netX, servingSide)
    CURRENT_DISPLAY = getDisplay(game.score, servingSide)
    print('Got serving side')

    gameMonitor = GameMonitor(game)

    while True:
        frame = view.read()

        # End of video
        if frame is None:
            print("Stream ended.")
            break

        if ball.updatePosFromFrame(prevFrame, frame, showProcessedFrame=False, showMaskFrame=False):
            ball.updateProcessedData(output=False)
            game.updateState(ball, output=False)

            gameMonitor.printNewEvents()

            if showDisplay:
                key = cv2.waitKey(int(1000 / view.fps * (slowDown if slowDown != -1 else 1.0)))
                if key == ord('q'):
                    print('Quiting.')
                    exit(0)
                if slowDown == -1:
                    input('>')

            if showFullDisplay:
                newDisplay = gameMonitor.getGameDisplay()
                if newDisplay is not None:
                    CURRENT_DISPLAY = newDisplay
                cv2.imshow('window', CURRENT_DISPLAY)
                if cv2.waitKey(1) == ord('q'):
                    break

        prevFrame = frame


def loadStream(res: tuple=(640,480), fps: int=30, loadVideo: Optional[str]=None, startFrame: Optional[int]=None) -> GameViewSource:
    # Load from video or from webcam
    if not loadVideo:
        stream = VideoStream(src=0, resolution=res, framerate=fps).start()
        time.sleep(2)
        fps = stream.stream.get(cv2.CAP_PROP_FPS)
        res = (int(stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        gameViewSource = GameViewSource(stream, False, res)
    else:
        stream = cv2.VideoCapture(loadVideo)
        fps = stream.get(cv2.CAP_PROP_FPS)
        if startFrame:
            stream.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        res = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        gameViewSource = GameViewSource(stream, True, res)
    print("Stream:\n\tFPS - %d\n\tResolution: (%d, %d)" % (fps, res[0], res[1]))

    # Setup net
    netX = userSetupScene(gameViewSource)
    gameViewSource.setNetPos(netX)
    print("Got net x: %d" % netX)

    # View has been setup
    return gameViewSource


if __name__ == '__main__':
    trackingArgs = setupArguments()
    trackingArgs['video'] = None
    trackingArgs['startFrame'] = 0
    trackingArgs['writeFrame'] = False
    trackingArgs['writeMasked'] = False
    trackingArgs['writeRaw'] = False
    trackingArgs['showProcessedFrame'] = False
    trackingArgs['showMaskFrame'] = False
    trackingArgs['show'] = False
    trackingArgs['game'] = True
    trackingArgs['debugWrite'] = False
    trackingArgs['slowDown'] = 1.0
    trackingArgs['fullDisplay'] = False
    view = loadStream()
    scoreGame(view, showFullDisplay=True)
