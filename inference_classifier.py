import cv2
import mediapipe as mp
import numpy as np
import pickle
from itertools import combinations

# Função para calcular distâncias 3D entre combinações de landmarks
def calculate_3d_distances(landmarks):
    distances = []
    for (i, j) in combinations(range(len(landmarks)//3), 2):
        x1, y1, z1 = landmarks[3*i], landmarks[3*i + 1], landmarks[3*i + 2]
        x2, y2, z2 = landmarks[3*j], landmarks[3*j + 1], landmarks[3*j + 2]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        distances.append(distance)
    return distances

# Carregar o modelo treinado
with open('rf_model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Inicializando MediaPipe e captura de vídeo
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Dicionário de rótulos
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 8: 'I', 11: 'L', 12: 'M'}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar o quadro.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                z_.append(landmark.z)

            if len(x_) != 21 or len(y_) != 21 or len(z_) != 21:
                print(f"Erro: Número incorreto de landmarks detectados. Detectado: {len(x_)} landmarks.")
                continue

            # Usa as coordenadas (x, y, z) dos landmarks
            combined_landmarks = []
            for i in range(21):
                combined_landmarks.append(x_[i])
                combined_landmarks.append(y_[i])
                combined_landmarks.append(z_[i])

            # Gerar as distâncias 3D
            distances = calculate_3d_distances(combined_landmarks)

            # Atualizando para 210 combinações
            if len(distances) != 210:
                print(f"Erro: O número de características geradas ({len(distances)}) não corresponde ao esperado (210).")
                continue

            # Convertendo as distâncias para numpy array
            data_aux = np.asarray(distances).astype(np.float32).reshape(1, -1)

            # Predição
            prediction = model.predict(data_aux)
            predicted_label = int(prediction[0])

            predicted_character = labels_dict.get(predicted_label, "Desconhecido")

            # Exibir resultado
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 2)

    cv2.imshow('Detecção de Sinais', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
