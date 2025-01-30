# Finger Tracking 3D Cube Rotation

This project uses **OpenCV** and **MediaPipe** to track hand movements and control the rotation of a 3D cube using your fingers.

## How It Works

1. **Tracks your fingers** – The program uses your webcam to detect your hand and track two fingers:  
   - **Thumb (tip)**  
   - **Index finger (tip)**  

2. **Measures finger distance** – The distance between your thumb and index finger is used to control rotation speed.

3. **Rotates a 3D cube** –  
   - Moving your **right hand's fingers** rotates the cube **left/right**.  
   - Moving your **left hand's fingers** rotates the cube **up/down**.  

4. **Displays two windows** –  
   - **"jarvis, track my fingers"** → Shows the webcam feed with finger tracking.  
   - **"cube"** → Displays the rotating 3D cube.  

## Controls

- Move your fingers closer or farther to change rotation speed.  
- Press **"q"** to exit the program.  

## Requirements

Make sure you have Python and the required libraries installed:

```bash
pip install opencv-python mediapipe numpy
```

## Work In Progress !
Alright now that added the gestures(which still can improve ofc)
I would like to keep expanding and testing the interaction with 3d spaces and objects

im gonna probably make a 3d engine to get a better undestanding of the application of matrices in projections
then ill come back to this

i will also add the missing code explanations to the read me, eventually...