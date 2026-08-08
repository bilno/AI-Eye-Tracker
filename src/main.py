import cv2
import mediapipe as mp
import time

from trackingdata import tracking_data
              
mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks = True)

LEFT_IRIS = [468, 469, 470, 471, 472]
LEFT_EYE_POINTS = [33, 133, 468, 469, 470, 471, 472]
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133

calibrating = True
start_time = None
ratio_list = []
center_ratio = None

def run_eye_tracker():
    global calibrating
    global start_time
    global ratio_list
    global center_ratio

    camera = cv2.VideoCapture(0)  

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
                # mp_drawing.draw_landmarks(
                #     image=frame,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION
                # )
                sumx = 0

                for landmark_index in LEFT_IRIS:
                    point = face_landmarks.landmark[landmark_index]
                    sumx += point.x


                for landmark_index in LEFT_EYE_POINTS:
                    point = face_landmarks.landmark[landmark_index]

                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])

                    cv2.circle(frame, (x,y), 3, (0, 255, 0), -1)                
                
                left_corner = face_landmarks.landmark[LEFT_EYE_LEFT_CORNER]
                right_corner = face_landmarks.landmark[LEFT_EYE_RIGHT_CORNER]
            
                average_x = sumx / len(LEFT_IRIS)

                eye_width = right_corner.x - left_corner.x
                iris_distance = average_x - left_corner.x
                ratio = iris_distance / eye_width

                tracking_data["ratio"] = ratio
                tracking_data["calibrating"] = calibrating

                if calibrating:
                    print("Calibrating... Please look straight ahead.")
                    if start_time is None:
                        start_time = time.time()

                    ratio_list.append(ratio)
                    elapsed = time.time() - start_time

                    # print("Elapsed:", elapsed) debugging

                    if elapsed >= 2:
                        center_ratio = sum(ratio_list) / len(ratio_list)
                        ratio_list.clear()

                        print("Calibration complete.")
                        
                        calibrating = False
                        tracking_data["calibrating"] = calibrating
                        tracking_data["center_ratio"] = center_ratio

                        # print("CALIBRATING DATA:", tracking_data["calibrating"]) debugging
                        print(f"Center ratio: {center_ratio:.3f}")

                else:
                    if ratio < center_ratio - 0.15:
                        direction = "Left"
                        tracking_data["direction"] = direction
                        print(f"Looking {direction}")


                    
                    elif ratio > center_ratio + 0.15:
                        direction = "Right"
                        tracking_data["direction"] = direction
                        print(f"Looking {direction}")
                    
                    else:
                        direction = "Center"
                        tracking_data["direction"] = direction
                        print(f"Looking {direction}")

                    print(f"Ratio: ({ratio:.3f})")

        cv2.imshow("Eye Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()