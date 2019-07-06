
from copy import deepcopy

# Import from main file
from Setup import LEFT, NET_VIEW_BUFFER, TABLE_END_BUFFER, display, getDisplay, getSideSignal, other, GameViewSource

# High level logic for the ping pong game
class GameState:

    STATE_PRE_SERVE = 0
    STATE_FREE_BALL = 1
    STATE_EXPECTING_RESPONSE = 2
    STATE_SERVE_ATTEMPT = 3
    STATE_AMBIGUOUS_BOUNCE = 4

    STATE_TO_NAME = {STATE_PRE_SERVE: 'Pre-Serve',
                     STATE_FREE_BALL: 'Free Ball',
                     STATE_EXPECTING_RESPONSE: 'Expecting Response',
                     STATE_AMBIGUOUS_BOUNCE: 'Ambiguous Bounce'}

    TIMEOUT_FRAMES_FOR_LONG_HIT = 20
    TIMEOUT_FRAMES_FOR_NO_HIT = 25

    def __init__(self, view: GameViewSource):
        self.state = self.STATE_PRE_SERVE
        # Constants
        self.netX = None
        # For display
        self.view = view
        # For pre-serve
        self.servingSide = None
        self.givenSecondTry = False
        self.serveCrossedNet = False
        # For expecting response
        self.expectingResponseFrom = None
        # For free ball
        self.freeBallFrom = None
        self.freeBallCrossedNet = False
        # For ambiguous bounce
        self.ambiguousBounceSide = None
        # Score keeping
        self.score = [0, 0]
        # Display
        self.currentDisplay = None

    def __copy__(self):
        obj = type(self)(self.view)
        obj.__dict__ = self.__dict__.copy()
        obj.__dict__['score'] = obj.__dict__['score'].copy()
        return obj

    def begin(self, netX, servingSide=None):
        self.state = self.STATE_PRE_SERVE
        self.netX = netX
        self.servingSide = servingSide
        self.givenSecondTry = False

    def transitionPreServe(self, servingSide):
        # Scoring
        if self.servingSide == servingSide:
            # print('%s HAS SCORED!' % display(servingSide))
            self.score[servingSide] += 1

        # Display
        self.currentDisplay = getDisplay(self.score, servingSide)

        # print('GAME: %s -> %s, serving from %s side' %
        #       (self.STATE_TO_NAME[self.state], self.STATE_TO_NAME[self.STATE_PRE_SERVE],
        #        display(servingSide)))

        self.state = self.STATE_PRE_SERVE
        self.servingSide = servingSide
        self.givenSecondTry = False
        self.serveCrossedNet = False

    def transitionExpectingResponse(self, expectingFrom):
        print('GAME: %s -> %s, expecting hit from %s side' %
              (self.STATE_TO_NAME[self.state], self.STATE_TO_NAME[self.STATE_EXPECTING_RESPONSE],
               display(expectingFrom)))

        self.state = self.STATE_EXPECTING_RESPONSE
        self.expectingResponseFrom = expectingFrom

    def transitionFreeBall(self, freeBallFrom):
        print('GAME: %s -> %s, ball hit from %s side' %
              (self.STATE_TO_NAME[self.state], self.STATE_TO_NAME[self.STATE_FREE_BALL],
               display(freeBallFrom)))

        self.state = self.STATE_FREE_BALL
        self.freeBallFrom = freeBallFrom
        self.freeBallCrossedNet = False

    def transitionAmbiguousBounce(self, ambiguousBounceSide):
        """An intermediate state that functions as either ER or FB."""
        print('GAME: %s -> %s, ambiguous bounce on the %s side' %
              (self.STATE_TO_NAME[self.state], self.STATE_TO_NAME[self.STATE_FREE_BALL],
               display(ambiguousBounceSide)))

        self.state = self.STATE_AMBIGUOUS_BOUNCE
        self.ambiguousBounceSide = ambiguousBounceSide

    def updateState(self, ball, output=False):
        if output:
            self.printCurrentState()

        # Pre Serve
        if self.state == self.STATE_PRE_SERVE:
            # Take note of instantaneous variables related to this state and store them
            if ball.ballCrossedTo == other(self.servingSide):
                if output: print('Serve crossed net')
                self.serveCrossedNet = True

            # Table bounce, Hit net, and Hit long requires the serve to have crossed the net
            if self.serveCrossedNet:
                # Table bounce
                if ball.bounceSide == other(self.servingSide):
                    if output: print('Table bounce')
                    self.transitionExpectingResponse(other(self.servingSide))
                # Hit net or hit long
                elif ball.hasHitNet or ball.framesOnSide > self.TIMEOUT_FRAMES_FOR_LONG_HIT:
                    # If this is the second time, change serving side but no point change
                    if self.givenSecondTry:
                        print('No more tries for you')
                        self.transitionPreServe(other(self.servingSide))
                    # Give a second try
                    else:
                        print('Giving a second try')
                        self.givenSecondTry = True
                        self.serveCrossedNet = False
                        self.state = self.STATE_PRE_SERVE

        # Free Ball
        elif self.state == self.STATE_FREE_BALL:
            # Error checking
            if self.freeBallFrom is None:
                raise RuntimeError('Need to know who hit this free ball')
            # Update variables
            if ball.ballCrossedTo == other(self.freeBallFrom):
                if output: print('Free ball crossed net')
                self.freeBallCrossedNet = True

            # Hit net
            if ball.hasHitNet:
                if output: print('Ball has hit the net')
                self.transitionPreServe(other(self.freeBallFrom))
            # Bounce early - I restricted the position of an early bounce because it almost always happens near the net
            elif ball.pos is not None and ball.pos[0] < self.netX + NET_VIEW_BUFFER and ball.bounceSide == self.freeBallFrom:
                print('Ball has bounced early')
                self.transitionPreServe(other(self.freeBallFrom))
            # Free ball has crossed the net - prerequisite for other side table bounce and hit long
            elif self.freeBallCrossedNet:
                # Table bounce on
                if ball.bounceSide == other(self.freeBallFrom):
                    # If the table bounce was within our confidence interval
                    if self.netX - TABLE_END_BUFFER < ball.lastPos[0] < self.netX + TABLE_END_BUFFER:
                        if output: print('Ball has bounced on the other side')
                        self.transitionExpectingResponse(other(self.freeBallFrom))
                    # If the bounce is near the edge of view it is treated as ambiguous
                    else:
                        if output: print('Ambiguous bounce')
                        self.transitionAmbiguousBounce(other(self.freeBallFrom))
                # Hit long
                elif ball.framesOnSide > self.TIMEOUT_FRAMES_FOR_LONG_HIT:
                    if output: print('Ball has been hit long')
                    self.transitionPreServe(other(self.freeBallFrom))
            # No hit
            elif ball.framesOnSide > self.TIMEOUT_FRAMES_FOR_NO_HIT:
                if output: print('Ball has not been hit')
                self.transitionPreServe(other(self.freeBallFrom))
            # Air hit on the other side
            elif ball.netSide == other(self.freeBallFrom) and ball.hitDirection == self.freeBallFrom:
                if output: print('Air hit')
                self.transitionExpectingResponse(other(self.freeBallFrom))

        # Expecting Response - Implied in this state is that the ball is on the returning player's side
        elif self.state == self.STATE_EXPECTING_RESPONSE:
            # Hit
            # if ball.hitDirection == other(self.expectingResponseFrom):
            isWithinReasonableBounds = self.netX - TABLE_END_BUFFER < ball.lastPos[0] < self.netX + TABLE_END_BUFFER
            isComingBack = ball.currentDir == other(self.expectingResponseFrom)
            if isComingBack and isWithinReasonableBounds:
                if output: print('Hit by player')
                self.transitionFreeBall(self.expectingResponseFrom)
            # Double bounce on responding side
            elif ball.bounceSide == self.expectingResponseFrom and isWithinReasonableBounds and ball.timeSinceBounce > 1:
                if output: print('Double bounce')
                self.transitionPreServe(other(self.expectingResponseFrom))
            # No hit - timeout
            elif ball.framesOnSide > self.TIMEOUT_FRAMES_FOR_NO_HIT:
                if output: print('Timeout hitting back the ball')
                self.transitionPreServe(other(self.expectingResponseFrom))

        # Ambiguous Bounce
        elif self.state == self.STATE_AMBIGUOUS_BOUNCE:
            # Timeout
            if ball.framesOnSide > self.TIMEOUT_FRAMES_FOR_LONG_HIT:
                print('Ambiguous bounce timeout - need to know who is serving')
                servingSide = getSideSignal(self.view, self.netX, self.score)
                self.transitionPreServe(servingSide)
            # Hit - removes the ambiguity and instantly changes state to free ball
            elif self.netX - NET_VIEW_BUFFER < ball.lastPos[0] < self.netX + NET_VIEW_BUFFER:
                print('Ambiguous bounce has been hit - now FB')
                self.transitionFreeBall(self.ambiguousBounceSide)

    def _hasGoneLong(self, ball, hasCrossed):
        """For shortening code."""
        return hasCrossed and ball.framesOnSide > self.TIMEOUT_FRAMES_FOR_LONG_HIT

    def printCurrentState(self):
        if self.state == self.STATE_PRE_SERVE:
            print('PS from the %s' % display(self.servingSide))
        elif self.state == self.STATE_FREE_BALL:
            print('FB from the %s' % display(self.freeBallFrom))
        elif self.state == self.STATE_EXPECTING_RESPONSE:
            print('ER from the %s' % display(self.expectingResponseFrom))
        elif self.state == self.STATE_AMBIGUOUS_BOUNCE:
            print('AB on the %s' % display(self.ambiguousBounceSide))

