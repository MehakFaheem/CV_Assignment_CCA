"""
Computer Vision Assignment: Pose Estimation & Activity Classification
Detects human poses, computes joint angles, and classifies activities.
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# TASK 1: POSE DETECTION & PRE-PROCESSING
_mp_pose_module = mp.solutions.pose
_pose_detector  = _mp_pose_module.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# HELPER FUNCTIONS
def calculate_angle(a, b, c):
    """Return the angle (degrees) at joint b formed by points a–b–c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    CONNECTIONS = [
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
    ]
    for s, e in CONNECTIONS:
        ls, le_ = landmarks[s], landmarks[e]
        if ls.visibility > 0.4 and le_.visibility > 0.4:
            p1 = (int(ls.x * w), int(ls.y * h))
            p2 = (int(le_.x * w), int(le_.y * h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for lm in landmarks:
        if lm.visibility > 0.4:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 0), -1)
    return frame

def smooth_keypoints(all_keypoints, window_length=5, polyorder=2):
    keypoints = np.array(all_keypoints)

    if len(keypoints) < window_length:
        return keypoints

    smoothed = np.copy(keypoints)

    for lm in range(keypoints.shape[1]):
        for coord in range(3):
            smoothed[:, lm, coord] = savgol_filter(
                keypoints[:, lm, coord],
                window_length=window_length,
                polyorder=polyorder
            )

    return smoothed

# TASK 2: JOINT ANGLE COMPUTATION & TRACKING
def compute_joint_angles(landmarks):
    def pt(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    return {
        'left_knee':      calculate_angle(pt(23), pt(25), pt(27)),
        'right_knee':     calculate_angle(pt(24), pt(26), pt(28)),
        'left_hip':       calculate_angle(pt(11), pt(23), pt(25)),
        'right_hip':      calculate_angle(pt(12), pt(24), pt(26)),
        'left_elbow':     calculate_angle(pt(11), pt(13), pt(15)),
        'right_elbow':    calculate_angle(pt(12), pt(14), pt(16)),
        'left_shoulder':  calculate_angle(pt(13), pt(11), pt(23)),
        'right_shoulder': calculate_angle(pt(14), pt(12), pt(24)),
    }

# TASK 3: RULE-BASED ACTIVITY CLASSIFICATION
_last_stable = 'SITTING'  

def classify_activity(angles):
    global _last_stable

    avg_knee     = (angles['left_knee']     + angles['right_knee'])     / 2
    avg_shoulder = (angles['left_shoulder'] + angles['right_shoulder']) / 2

    if avg_knee < 115:
        activity = 'ARMS_RAISED_SITTING' if avg_shoulder >= 50 else 'SITTING'
        _last_stable = activity
        return activity

    elif avg_knee >= 165:
        activity = 'ARMS_RAISED_STANDING' if avg_shoulder >= 50 else 'STANDING'
        _last_stable = activity
        return activity

    else:
        if _last_stable in ('STANDING', 'ARMS_RAISED_STANDING'):
            return 'SQUATTING'
        return _last_stable

# MAIN VIDEO-PROCESSING FUNCTION
ACTIVITY_COLOURS = {
    'SITTING':              (255, 128,   0),
    'ARMS_RAISED_SITTING':  (255, 200,   0),
    'ARMS_RAISED_STANDING': (180,   0, 255),
    'SQUATTING':            (  0, 200, 255),
    'STANDING':             (  0, 255,   0),
}

def process_video(video_source, output_video_path='output_pose.mp4', manual_labels=None):
    global _last_stable
    _last_stable = 'SITTING'   

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_source}")
        return None

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}×{height}, {fps} FPS, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    all_keypoints   = []
    all_angles      = defaultdict(list)
    all_predictions = []
    frame_count     = 0

    print("Processing frames …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = _pose_detector.process(frame_rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            kp = np.array([[lm.x, lm.y, lm.z] for lm in lms])
            all_keypoints.append(kp)

            angles   = compute_joint_angles(lms)
            activity = classify_activity(angles)

            for joint, val in angles.items():
                all_angles[joint].append(val)
            all_predictions.append(activity)

            frame = draw_skeleton(frame, lms)

            avg_knee     = (angles['left_knee']     + angles['right_knee'])     / 2
            avg_shoulder = (angles['left_shoulder'] + angles['right_shoulder']) / 2
            colour       = ACTIVITY_COLOURS.get(activity, (200, 200, 200))

            panel_width = 360
            panel_height = 130
            panel_x = width - panel_width - 10
            panel_y = 30
            
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (20, 20, 20), -1)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (100, 100, 100), 3)
            
            cv2.putText(frame, f'Activity: {activity}', (panel_x + 12, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(frame, f'Knee:     {avg_knee:.1f} deg', (panel_x + 12, panel_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f'Shoulder: {avg_shoulder:.1f} deg', (panel_x + 12, panel_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f'Frame: {frame_count}', (panel_x + 12, panel_y + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            all_predictions.append(_last_stable)
            
            panel_width = 350
            panel_height = 130
            panel_x = width - panel_width - 10
            panel_y = 10
            
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (20, 20, 20), -1)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (100, 100, 100), 2)
            cv2.putText(frame, 'No pose detected', (panel_x + 30, panel_y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 60 == 0:
            print(f"  {frame_count}/{total} frames processed …")

    cap.release()
    out.release()
    print("Smoothing keypoints …")
    smoothed = smooth_keypoints(all_keypoints, window_length=5)
    print("Smoothing applied successfully.")

    # Accuracy evaluation
    accuracy = None
    if manual_labels:
        correct = sum(
            1 for frame_num, gt in manual_labels.items()
            if frame_num < len(all_predictions) and all_predictions[frame_num] == gt
        )
        accuracy = correct / len(manual_labels) * 100
        print(f"Classification Accuracy: {accuracy:.2f}%")

    return {
        'keypoints':          all_keypoints,
        'smoothed_keypoints': smoothed,
        'angles':             all_angles,
        'predictions':        all_predictions,
        'accuracy':           accuracy,
        'total_frames':       frame_count,
        'fps':                fps,
    }

# VISUALISATION
def plot_joint_angles(results, save_path='joint_angles.png'):
    angles = results['angles']

    predictions = results['predictions']
    transitions = [i for i in range(1, len(predictions))
                   if predictions[i] != predictions[i-1]]

    joints = [
        ('left_shoulder',  'Left Shoulder Elevation'),
        ('right_shoulder', 'Right Shoulder Elevation'),
        ('left_elbow',     'Left Elbow Angle'),
        ('right_elbow',    'Right Elbow Angle'),
        ('left_knee',      'Left Knee Angle'),
        ('right_knee',     'Right Knee Angle'),
        ('left_hip',       'Left Hip Angle'),
        ('right_hip',      'Right Hip Angle'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(15, 14))
    fig.suptitle('Joint Angles Over Time', fontsize=16)

    for idx, (joint, title) in enumerate(joints):
        ax = axes[idx // 2, idx % 2]

        ax.plot(angles[joint], linewidth=2)

        # Activity transition markers
        for t in transitions:
            if t < len(angles[joint]):
                ax.axvline(x=t, color='red',
                           linestyle='--',
                           alpha=0.4,
                           linewidth=1)

        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (°)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_activity_classification(results, save_path='activity_timeline.png'):
    predictions = results['predictions']
    activity_map = {
        'SITTING':              1,
        'ARMS_RAISED_SITTING':  2,
        'SQUATTING':            3,
        'STANDING':             4,
        'ARMS_RAISED_STANDING': 5,
    }
    colours_map = {
        1: '#FF8000',
        2: '#FFC800',
        3: '#00C8FF',
        4: '#00FF00',
        5: '#B400FF',
    }
    numeric = [activity_map.get(p, 0) for p in predictions]
    fps     = results.get('fps', 30)
    times   = [i / fps for i in range(len(numeric))]

    fig, ax = plt.subplots(figsize=(16, 5))
    for i in range(len(numeric) - 1):
        ax.fill_between([times[i], times[i+1]], [numeric[i], numeric[i+1]],
                        color=colours_map.get(numeric[i], '#888888'), alpha=0.6, step='post')
    ax.step(times, numeric, linewidth=1.5, where='post', color='black', alpha=0.6)

    ax.set_title('Activity Classification Over Time', fontsize=14)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activity')
    ax.set_yticks([1, 2, 3, 4, 5])                                         
    ax.set_yticklabels(['SITTING', 'ARMS_RAISED_SITTING', 'SQUATTING',
                        'STANDING', 'ARMS_RAISED_STANDING'])                
    ax.set_ylim(0.5, 5.5)                                                   
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF8000', label='SITTING'),
        Patch(facecolor='#FFC800', label='ARMS_RAISED_SITTING'),
        Patch(facecolor='#00C8FF', label='SQUATTING'),
        Patch(facecolor='#00FF00', label='STANDING'),
        Patch(facecolor='#B400FF', label='ARMS_RAISED_STANDING'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("POSE ESTIMATION & ACTIVITY CLASSIFICATION")
    print("=" * 70)

    video_source = 'video.mp4'

    manual_labels = {
        0:   'SITTING',
        30:  'SITTING',
        50:  'SITTING',
        80:  'ARMS_RAISED_SITTING',
        100: 'ARMS_RAISED_SITTING',
        130: 'ARMS_RAISED_SITTING',
        185: 'ARMS_RAISED_STANDING',
        230: 'ARMS_RAISED_STANDING',
        270: 'ARMS_RAISED_STANDING',
        310: 'ARMS_RAISED_STANDING', 
        340: 'ARMS_RAISED_SITTING',
        365: 'ARMS_RAISED_SITTING',
        400: 'STANDING',
        450: 'STANDING',
        530: 'STANDING',
        590: 'SQUATTING',
        610: 'SQUATTING',
        655: 'STANDING',
        677: 'STANDING',
    }

    try:
        results = process_video(
            video_source,
            output_video_path='output_pose.mp4',
            manual_labels=manual_labels,
        )

        if results:
            print(f"\nProcessed {results['total_frames']} frames")
            print(f"Predictions made: {len(results['predictions'])}")
            if results['accuracy'] is not None:
                print(f"Accuracy: {results['accuracy']:.2f}%")

            plot_joint_angles(results)
            plot_activity_classification(results)

            print("\n" + "=" * 70)
            print("Saved Output files:")
            print("  output_pose.mp4")
            print("  joint_angles.png")
            print("  activity_timeline.png")
            print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()