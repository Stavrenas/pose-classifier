
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:24:36 2022

@author: user
"""
import csv
import mediapipe as mp
import cv2

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

filename = 'test_video'
videp_file = filename + '.mp4'
cap = cv2.VideoCapture(videp_file)

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out_filename = 'output.mp4'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
    'M', 'P', '4', 'V'), fps, (frame_width, frame_height))

csv_name = filename + '.csv'
file = open(csv_name, 'w', newline='')
header = 'nose_x nose_y left_eye_inner_x left_eye_inner_y left_eye_x left_eye_y left_eye_outer_x left_eye_outer_y \
        right_eye_inner_x right_eye_inner_y right_eye_x right_eye_y right_eye_outer_x right_eye_outer_y left_ear_x left_ear_y \
        right_ear_x right_ear_y mouth_left_x mouth_left_y mouth_right_x mouth_right_y left_shoulder_x left_shoulder_y \
        right_shoulder_x right_shoulder_y left_elbow_x left_elbow_y right_elbow_x right_elbow_y left_wrist_x left_wrist_y \
        right_wrist_x right_wrist_y left_pinky_x left_pinky_y right_pinky_x right_pinky_y left_index_x left_index_y \
        right_index_x right_index_y left_thumb_x left_thumb_y right_thumb_x right_thumb_y left_hip_x left_hip_y right_hip_x right_hip_y \
        left_knee_x left_knee_y right_knee_x right_knee_y left_ankle_x left_ankle_y right_ankle_x right_ankle_y left_heel_x left_heel_y \
        right_heel_x right_heel_y left_foot_index_x left_foot_index_y right_foot_index_x right_foot_index_y'
header = header.split()
with file:
    writer = csv.writer(file)
    writer.writerow(header)

all_data = []

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('MediaPipe Pose', image)
        out.write(image)
        if results.pose_landmarks:
            to_append = ''
            for i in range(int(len(header)/2)):
                # =============================================================================
                #                 results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * image.shape[0]
                #                 results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * image.shape[1]
                # =============================================================================
                to_append += f'{results.pose_landmarks.landmark[i].x} {results.pose_landmarks.landmark[i].y} '
            file = open(csv_name, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
out.release()
cv2.destroyAllWindows()
