# PingPongScoring Overview
This script follows a game of ping pong by using computer vision techniques and a state machine. This project was inspired by my curiosity of how accurately a computer algorithm could score a ping pong game - I wanted to make a proof-of-concept. (Oh the things one thinks about while bored at home!) The steps of the project were to 1) identify the ball 2) pinpoint the ball's location and 3) track the game mechanics. 

To start on identifying the ball, I knew my best bet was to use the distinct hue of the neon yellow ball as the ball's identifying factor. I grabbed my laptop, webcam, and camera tripod and started collecting footage of me hitting that ping pong ball across the table at various speeds. At slow speeds, my webcam was able to capture a defined ball shape, but at high speeds, the motion blur of the camera turned the ball into a yellow smear across each frame. I knew I needed a more intelligent solution than just to identify the position of the ball by color, so I opted for a guessing approach. `Ball.py` contains the code I wrote to intelligently narrow down the location of the ball despite motion blur.

Here is an overview of the contained files:

### Project Structure

- *PingPongDetector.py* - Top level file containing OpenCV code for drawing the GUI display, loading the video stream, and the main game scoring loop.
- *Game.py* - Contains the game state machine and the conditions for transitioning between states. Probabilistic logic is done in this file to account for some uncertainty with the ball's position.
- *Ball.py* - Contains OpenCV and Numpy code to update the location of the ball (x and y coordinates on screen) given a new frame from the video stream. Most OpenCV code is found here.

## Example Tracking GIF

![Ball Tracking](https://im3.ezgif.com/tmp/ezgif-3-624e790ff253.gif)
