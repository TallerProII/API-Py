import base64
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image
from io import BytesIO

def funcion(cod_base64: str):
    mensaje = " "
    text = " "
    try:
        # region IMAGEN BASE 64 - A MATRIZ NUMPY
        # Cadena de caracteres codificada en base64
        base64_image = cod_base64

        # Decodificar la cadena de caracteres en base64 a bytes
        image_bytes = base64.b64decode(base64_image)

        # Crear un objeto Image a partir de los bytes decodificados
        image = Image.open(BytesIO(image_bytes))

        # Convertir la imagen PIL en una matriz NumPy
        image_np = np.array(image)
        # endregion

        model_dict = pickle.load(open('./model2.p', 'rb'))
        model = model_dict['model']

        labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
                       9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
                       18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

        data = []
        labels = []
        processed_images = 0
        data_aux = []
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Obtener las dimensiones de la imagen
        H, W, _ = image_np.shape

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                #region AJUSTAR COORDEDNDAS TOMANDO DE REFERENCIA EL PUNTO 9
                pos9_x = hand_landmarks.landmark[9].x
                pos9_y = hand_landmarks.landmark[9].y
                for i in range(len(hand_landmarks.landmark)):
                    # for i in range(len(hand_landmarks.landmark)):
                    pos_x = hand_landmarks.landmark[i].x
                    pos_y = hand_landmarks.landmark[i].y
                    x1_ = (pos_x - (pos9_x - 0.5))  # 0.5 = (m_ancho/ancho_)
                    y1_ = (pos_y - (pos9_y - 0.5))
                    data_aux.append(x1_)
                    data_aux.append(y1_)
                #endregion

                # region PREDICCON DE LA IMAGEN
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                confidence = model.predict_proba([np.asarray(data_aux)])[0].max() * 100

                # img_path_full = os.path.join(DATA_DIR, dir_, img_path)
                text = f'{predicted_character} ({confidence:.2f}%)'
                # endregion

        #region MOSTRAR IMAGEN
        # cv2.imshow('frame', img_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #endregion

    except Exception as e:
        mensaje = e
        pass
    return {"text": text, "mensaje" : mensaje}
