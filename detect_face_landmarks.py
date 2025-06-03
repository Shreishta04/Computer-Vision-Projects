import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#create object of VideoCapture
cap = cv2.VideoCapture(0) # 0 is the default cam

while True:
    #to capture the image we use read() and it returns key and img
    key,img = cap.read()

    # objects captured using cv is in BGR, we use cvtColor to convert this BGR to RGB
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #face landmarks will be stored in results
    results = face_mesh.process(rgb_frame)

    #multiple landmarks on the face will be shown
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for point in landmarks.landmark:
                x, y = int(point.x * img.shape[1]), int(point.y * img.shape[0])
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    resized_img = cv2.resize(img, (480, 360))
    #display the image in a window
    cv2.imshow("my_video",resized_img)
    #it constantly captures image by image so we wait a second to keep it from getting stuck
    cv2.waitKey(1)