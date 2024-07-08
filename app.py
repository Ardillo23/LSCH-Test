from flask import Flask, render_template, Response, jsonify, send_file
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

# print("Número de acciones:", len(actions))

sequence = []
sentence = []
threshold = 0.98 #  Umbral de confianza para considerar una acción reconocida. / tenia 0.8
camera_active = False
last_prediction_time = time.time()
prediction_interval = 2  #  Variables para controlar el intervalo entre predicciones / Tenia 2

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    max_prob_index = np.argmax(res)
    max_prob = res[max_prob_index]
    
    # Verificar que el índice esté dentro del rango
    if max_prob_index < len(actions):
        action = actions[max_prob_index]
    else:
        action = "Unknown"

    # Draw rectangle for background (static)
    cv2.rectangle(output_frame, (0, 0), (640, 40), (50, 50, 255), -1)
    
    # Display only the probability on the right side of the blue rectangle
    prob_text = f"{max_prob * 100:.2f}%"
    text_size, _ = cv2.getTextSize(prob_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = 640 - text_size[0] - 10  # Right align with a margin of 10 pixels
    text_y = 40 - (40 - text_size[1]) // 2  # Center vertically
    cv2.putText(output_frame, prob_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output_frame


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def gen():
    global sequence, sentence, camera_active

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            if camera_active:
                ret, frame = cap.read()
                if not ret:
                    break

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
                        sentence = sentence[-1:]  # Keep only the latest prediction

                    image = prob_viz(res, actions, image)

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
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    # app.run(debug=True) #esto hace que corra solo en el computador
    app.run(host='0.0.0.0', port=5000, debug=True) #Esto hace que corra en todos los dispositivos de la casa
