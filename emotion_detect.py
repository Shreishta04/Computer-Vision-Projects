import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()

    results = DeepFace.analyze(img, actions=["emotion"],enforce_detection = False)

    emotion = results[0]['dominant_emotion']

    #text, org, font,font scale,color,thickness
    cv2.putText(img, f'Emotion: {emotion}',(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    resized_img = cv2.resize(img, (480, 360))
    cv2.imshow("Emotion Recognition",resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

