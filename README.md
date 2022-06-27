# Virtual Mouse - Computer Vision 
Control cursor and click mouse using your eyes and your laptop's webcam

### Install

This project files requires **Python 3** and the following Python libraries installed:

- [OpenCV](https://opencv.org/)
- [Numpy](http://numpy.org/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)


### Run

```bash
python3 faceMesh.py
```  

Plug in a webcam and run python faceMesh.py. It will show the webcam image. If it sees your iris, it will mark the centre of them and move cursor according to where you look. 

If you close your left/right eye it will respectfully left/right click.
