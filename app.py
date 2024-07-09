from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp
import time

app = Flask(__name__)

model = load_model('C:/Users/kevin/OneDrive/one drive/Escritorio/Proyecto/model/Señas6.h5')
mp_holistic = mp.solutions.holistic
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
colores = np.array(['Rojo', 'Azul', 'Amarillo', 'Verde', 'Morado', 'Rosado', 'Cafe', 'Blanco', 'Negro', 'Gris', 'Naranjo'])
preguntas = np.array(['Como', 'Como estas', 'Cual', 'Cuando', 'Cuantos', 'Donde', 'Porque', 'Que', 'Que paso'])
saludos = np.array(['Bienvenido', 'Buenas noches', 'Buenas tardes', 'Buenos dias', 'Chao', 'Gracias', 'Hola', 'Por favor'])
Pronombres = np.array(['Yo', 'Tu', 'El', 'Todos'])
auxiliar = np.concatenate((colores, preguntas), axis=None)
auxiliar2 = np.concatenate((auxiliar, saludos), axis=None)
actions = np.concatenate((auxiliar2, Pronombres), axis=None)

sequence = []
sentence = []
threshold = 0.98
camera_active = False
last_prediction_time = time.time()
prediction_interval = 2

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, frame_width, frame_height):
    output_frame = input_frame.copy()
    max_prob_index = np.argmax(res)
    max_prob = res[max_prob_index]
    
    if max_prob_index < len(actions):
        action = actions[max_prob_index]
    else:
        action = "Unknown"
    
    # Define la posición para el rectángulo y el texto en la parte inferior izquierda
    rect_width = 250
    rect_height = 40
    start_x = 10  # Coordenada x inicial del rectángulo
    start_y = frame_height - rect_height - 10  # Coordenada y inicial del rectángulo
    end_x = start_x + rect_width  # Coordenada x final del rectángulo
    end_y = frame_height - 10  # Coordenada y final del rectángulo
    
    # Dibuja el rectángulo
    cv2.rectangle(output_frame, (start_x, start_y), (end_x, end_y), (50, 50, 255), -1)
    
    # Texto adicional "Porcentaje de probabilidad" (encima del recuadro)
    additional_text = "Porcentaje de probabilidad"
    additional_text_x = start_x + 10
    additional_text_y = start_y - 10  # Posiciona este texto justo encima del rectángulo
    cv2.putText(output_frame, additional_text, (additional_text_x, additional_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Texto de probabilidad
    prob_text = f"{action}: {max_prob * 100:.2f}%"
    text_x = start_x + 10
    text_y = start_y + rect_height - 10
    cv2.putText(output_frame, prob_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return output_frame





def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def gen(device_type):
    global sequence, sentence, camera_active

    if device_type == "mobile":
        cap = cv2.VideoCapture('http://192.168.0.2:4747/video')  # URL del stream del teléfono
    else:
        cap = cv2.VideoCapture(0)  # Cámara del PC

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            if camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_height, frame_width = frame.shape[:2]

                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    max_prob_index = np.argmax(res)

                    if max_prob_index < len(actions) and res[max_prob_index] > threshold:
                        action = actions[max_prob_index]
                        sentence.append(action)
                        sentence = sentence[-1:]

                    image = prob_viz(res, actions, image, frame_width, frame_height)

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                frame = cv2.imread('static/imagen/placeholder.jpg')
                if frame is not None:
                    frame = cv2.resize(frame, (650, 480), interpolation=cv2.INTER_AREA)
                else:
                    frame = np.zeros((650, 480, 3), np.uint8)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobi' in user_agent:
        device_type = "mobile"
    else:
        device_type = "pc"
    return Response(gen(device_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify(success=True)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
