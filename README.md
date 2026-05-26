# Pose Estimation and Activity Classification using MediaPipe Pose

This project implements a Computer Vision pipeline for human pose estimation and activity classification using MediaPipe Pose, OpenCV, and Python.

The system detects human body landmarks from a video, computes important joint angles, and classifies activities using a rule-based classifier.

## Features

* Human pose detection using MediaPipe Pose
* Skeleton visualization on video frames
* Joint angle computation
* Rule-based activity classification
* Savitzky-Golay smoothing for reducing pose jitter
* Joint angle plotting
* Activity timeline visualization
* Output annotated video generation

## Activities Detected

* SITTING
* STANDING
* ARMS_RAISED_STANDING
* ARMS_RAISED_SITTING
* SQUATTING

## Libraries Used/Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* SciPy
* Matplotlib

## Project Workflow

1. Load input video
2. Detect body landmarks using MediaPipe Pose
3. Compute joint angles
4. Apply smoothing on keypoint coordinates
5. Classify activities using angle thresholds
6. Generate visualizations and output video

## Output Files

* `output_pose.mp4` — Annotated output video
* `joint_angles.png` — Joint angle plots
* `activity_timeline.png` — Activity classification timeline

## Accuracy

The system achieved an activity classification accuracy of 94.74% on manually labeled video frames.

## Video Source

[https://youtu.be/ITv-_BkcrD0?si=tr1N1beYWCool6QP](https://youtu.be/ITv-_BkcrD0?si=tr1N1beYWCool6QP)

## How to Run

Install required libraries:

```bash
pip install opencv-python mediapipe numpy scipy matplotlib
```

Run the project:

```bash
python pose_activity_classifier.py
```

Mehak Faheem

BS Artificial Intelligence

6th Semester
