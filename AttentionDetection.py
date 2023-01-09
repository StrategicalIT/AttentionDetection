import cv2 
import mediapipe as mp
import math
import numpy as np 
import os 
import time

# function for calculating euclidean distance
def euclidean_distance(pt1, pt2):

    return (((pt1[:2] - pt2[:2])**2).sum())**0.5

# function for calculating EAR
def eye_aspect_ratio(landmarks):

    lefti1 = euclidean_distance(landmarks[left_eye[1][0]], landmarks[left_eye[1][1]])
    lefti2 = euclidean_distance(landmarks[left_eye[2][0]], landmarks[left_eye[2][1]])
    lefti3 = euclidean_distance(landmarks[left_eye[3][0]], landmarks[left_eye[3][1]])
    DistLeft = euclidean_distance(landmarks[left_eye[0][0]], landmarks[left_eye[0][1]])
    lefti= (lefti1 + lefti2 + lefti3) / (3 * DistLeft)

    righti1 = euclidean_distance(landmarks[right_eye[1][0]], landmarks[right_eye[1][1]])
    righti2 = euclidean_distance(landmarks[right_eye[2][0]], landmarks[right_eye[2][1]])
    righti3 = euclidean_distance(landmarks[right_eye[3][0]], landmarks[right_eye[3][1]])
    DistRight = euclidean_distance(landmarks[right_eye[0][0]], landmarks[right_eye[0][1]])
    righti= (righti1 + righti2 + righti3) / (3 * DistRight)

    return (lefti + righti) / 2

# function for calculating MAR
def mouth_aspect_ratio(landmarks):

    mouth1 = euclidean_distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    mouth2 = euclidean_distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    mouth3 = euclidean_distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    DistMouth = euclidean_distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (mouth1 + mouth2 + mouth3)/(3*DistMouth)

# function for calculating Pupil centricity
def pupil_circularity(landmarks):
    LeftPeri = euclidean_distance(landmarks[left_eye[0][0]], landmarks[left_eye[1][0]]) + \
            euclidean_distance(landmarks[left_eye[1][0]], landmarks[left_eye[2][0]]) + \
            euclidean_distance(landmarks[left_eye[2][0]], landmarks[left_eye[3][0]]) + \
            euclidean_distance(landmarks[left_eye[3][0]], landmarks[left_eye[0][1]]) + \
            euclidean_distance(landmarks[left_eye[0][1]], landmarks[left_eye[3][1]]) + \
            euclidean_distance(landmarks[left_eye[3][1]], landmarks[left_eye[2][1]]) + \
            euclidean_distance(landmarks[left_eye[2][1]], landmarks[left_eye[1][1]]) + \
            euclidean_distance(landmarks[left_eye[1][1]], landmarks[left_eye[0][0]])
    leftArea = math.pi * ((euclidean_distance(landmarks[left_eye[1][0]], landmarks[left_eye[3][1]]) * 0.5) ** 2)

    RightPeri = euclidean_distance(landmarks[right_eye[0][0]], landmarks[right_eye[1][0]]) + \
               euclidean_distance(landmarks[right_eye[1][0]], landmarks[right_eye[2][0]]) + \
               euclidean_distance(landmarks[right_eye[2][0]], landmarks[right_eye[3][0]]) + \
               euclidean_distance(landmarks[right_eye[3][0]], landmarks[right_eye[0][1]]) + \
               euclidean_distance(landmarks[right_eye[0][1]], landmarks[right_eye[3][1]]) + \
               euclidean_distance(landmarks[right_eye[3][1]], landmarks[right_eye[2][1]]) + \
               euclidean_distance(landmarks[right_eye[2][1]], landmarks[right_eye[1][1]]) + \
               euclidean_distance(landmarks[right_eye[1][1]], landmarks[right_eye[0][0]])
    rightArea = math.pi * ((euclidean_distance(landmarks[right_eye[1][0]], landmarks[right_eye[3][1]]) * 0.5) ** 2)

    return (4*math.pi*leftArea)/(LeftPeri**2) + (4*math.pi*rightArea)/(RightPeri**2) / 2

#landmark points defined
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]] # mouth landmark coordinates

# Declaring FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
drawing_spec = mp_drawing.DrawingSpec(0,1,1)

label = None

input_data = []
frame_before_run = 0
color = (0, 0, 255)

#capture video
#cap = cv2.VideoCapture(0) #use this line if capture video from a webcam
cap = cv2.VideoCapture("video01.mp4") #use this line to read a video file
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        #continue #use this line for webcam capture
        break #use this line if video file read
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    #pass the frame through facemesh to get the landmarks
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append(
                [data_point.x, data_point.y, data_point.z])  # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # draw face mesh over image
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        ear = eye_aspect_ratio(landmarks_positions)
        mar = mouth_aspect_ratio(landmarks_positions)
        puc = pupil_circularity(landmarks_positions)

    else:
        ear = -1000
        mar = -1000
        puc = -1000

    if len(input_data) == 20:
        input_data.pop(0)
    input_data.append([ear, mar, puc])

    frame_before_run += 1
    if frame_before_run >= 15 and len(input_data) == 20:
        frame_before_run = 0

        if ear >= 0.28 and mar <= 0.8:
            label = 'Attentive'
            color = (0, 255, 0)
        else:
            label = 'Distracted'
            color = (0, 0, 255)
   
    cv2.putText(image, "%s" % label, (int(0.02 * image.shape[1]), int(0.2 * image.shape[0])),cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    cv2.imshow('Student Attention Detection', image)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

face_mesh.close()
