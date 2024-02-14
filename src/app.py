import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dis
import threading, sys, time
import pyttsx3

from src.utils import draw_landmarks, run_speech, get_aspect_ratio, parameters

# Initialize TTS engine
speech_engine = pyttsx3.init()
speech_lock = threading.Lock()  # Create a lock for synchronization

class drawsi:
    def __init__(self):
        self.frame_count = 0
        self.min_frame = 6
        self.min_tolerance = 5.0

    def face_cal(self):
        try:
            face_mesh = mp.solutions.face_mesh
            draw_utils = mp.solutions.drawing_utils
            landmark_style = draw_utils.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
            connection_style = draw_utils.DrawingSpec((0,0,255), thickness=1, circle_radius=1)

            face_model = face_mesh.FaceMesh(static_image_mode=parameters.STATIC_IMAGE,
                                            max_num_faces=parameters.MAX_NO_FACES,
                                            min_detection_confidence=parameters.DETECTION_CONFIDENCE,
                                            min_tracking_confidence=parameters.TRACKING_CONFIDENCE)
            return face_model
        except Exception as e:
            print(e, sys)

    def main_fun(self, cap, face_model):
        try:
            start_time = time.time()
            while True:
                result, image = cap.read()
                if result:
                    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    outputs = face_model.process(image_rgb)

                    if outputs.multi_face_landmarks:        
                        draw_landmarks(image, outputs, parameters.LEFT_EYE_TOP_BOTTOM, parameters.COLOR_RED)
                        draw_landmarks(image, outputs, parameters.LEFT_EYE_LEFT_RIGHT, parameters.COLOR_RED)
                        
                        ratio_left =  get_aspect_ratio(image, outputs, parameters.LEFT_EYE_TOP_BOTTOM, parameters.LEFT_EYE_LEFT_RIGHT)
                        
                
                        draw_landmarks(image, outputs, parameters.RIGHT_EYE_TOP_BOTTOM, parameters.COLOR_RED)
                        draw_landmarks(image, outputs, parameters.RIGHT_EYE_LEFT_RIGHT, parameters.COLOR_RED)
                        
                        ratio_right =  get_aspect_ratio(image, outputs, parameters.RIGHT_EYE_TOP_BOTTOM, parameters.RIGHT_EYE_LEFT_RIGHT)
                        
                        ratio = (ratio_left + ratio_right)/2.0

                        if ratio > self.min_tolerance:
                            elapsed_time = int(time.time() - start_time)
                            cv.putText(image, f"Time: {elapsed_time}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                            
                            if elapsed_time > 3:
                                with speech_lock:
                                    self.frame_count += 1
                        else:
                            self.frame_count = 0
                            
                        if self.frame_count > self.min_frame:
                            message = 'Drowsy Alert: It Seems you are sleeping.. please wake up'
                            elapsed_time = 0
                            with speech_lock:
                                t = threading.Thread(target=run_speech, args=(speech_engine, message))
                                t.start()

                        draw_landmarks(image, outputs, parameters.UPPER_LOWER_LIPS , parameters.COLOR_BLUE)
                        draw_landmarks(image, outputs, parameters.LEFT_RIGHT_LIPS, parameters.COLOR_BLUE)
                        
                        
                        ratio_lips =  get_aspect_ratio(image, outputs, parameters.UPPER_LOWER_LIPS, parameters.LEFT_RIGHT_LIPS)
                        if ratio_lips < 1.8:
                            message = 'Drowsy Warning: You looks tired.. please take rest'
                            with speech_lock:
                                p = threading.Thread(target=run_speech, args=(speech_engine, message))
                                p.start()

                    cv.imshow("FACE MESH", image)
                    if cv.waitKey(5) & 0xFF == ord('q'):
                        break

        except Exception as e:
            print(e)
        finally:
            cap.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    obj = drawsi()

    Fmodel = obj.face_cal()
    obj.main_fun(cv.VideoCapture(0), Fmodel)
