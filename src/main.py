import cv2
import mediapipe as mp
import time

camera = cv2.VideoCapture(0)                

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks = True)

LEFT_IRIS = [468, 469, 470, 471, 472]
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133

calibrating = True
start_time = None
ratio_list = []
center_ratio = None

if not camera.isOpened():                       
    print("Error: Could not open webcam.")
    exit()

while True:                                    
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)

    if not success:
        print("Error: Could not read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION
            )
            sumx = 0
            #sumy = 0

            for landmark_index in LEFT_IRIS:
                point = face_landmarks.landmark[landmark_index]
                sumx += point.x
                #sumy += point.y
            
            left_corner = face_landmarks.landmark[LEFT_EYE_LEFT_CORNER]
            right_corner = face_landmarks.landmark[LEFT_EYE_RIGHT_CORNER]
        
            average_x = sumx / len(LEFT_IRIS)
            #average_y = sumy / len(LEFT_IRIS)

            eye_width = right_corner.x - left_corner.x
            iris_distance = average_x - left_corner.x
            ratio = iris_distance / eye_width

            if calibrating:
                print("Calibrating... Please look straight ahead.")
                if start_time is None:
                    start_time = time.time()

                ratio_list.append(ratio)
                elapsed = time.time() - start_time

                if elapsed >= 2:
                    center_ratio = sum(ratio_list) / len(ratio_list)
                    ratio_list.clear()
                    print("Calibration complete.")
                    print(f"Center ratio: {center_ratio:.3f}")
                    calibrating = False

            else:
                if ratio < center_ratio - 0.15:
                    print("Looking Left")
                
                elif ratio > center_ratio + 0.15:
                    print("Looking Right")
                
                else:
                    print("Looking Center")

                print(f"Ratio: ({ratio:.3f})")

    cv2.imshow("Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()