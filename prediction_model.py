import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import shutil
from colorama import Fore, Style
try:
    model_dict = pickle.load(open('./model2.p', 'rb'))
    model = model_dict['model']

    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
                   9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
                   18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)  # Limit to 2 hands

    DATA_DIR = './data2'

    data = []
    labels = []
    processed_images = 0
    for dir_ in sorted(os.listdir(DATA_DIR), key=lambda x: int(x)):
        count = 0
        aux=0
        porc=[]
        processed_images = int(dir_)
        print('\t Directorio {} - {}'.format(processed_images,labels_dict[processed_images]))
        for img_path in sorted(os.listdir(os.path.join(DATA_DIR, dir_)), key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, dir_, x))):
            # if count >= 1:
            #    break
            data_aux = []

            x_ = []
            y_ = []
            z_ = []

            frame = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow('frame', frame)

            # Obtener las dimensiones de la imagen
            # H, W, _ = frame.shape

            results = hands.process(frame)
            if results.multi_hand_landmarks:
                # print("Se detectó una mano")
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                    # data_aux = []
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
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    prediction = model.predict([np.asarray(data_aux)])

                    predicted_character = labels_dict[int(prediction[0])]
                    confidence = model.predict_proba([np.asarray(data_aux)])[0].max() * 100


                    img_path_full = os.path.join(DATA_DIR, dir_, img_path)
                    text = f'{img_path_full}\t{predicted_character} ({confidence:.2f}%)'

                    true_label = labels_dict[processed_images]
                    if true_label != predicted_character:
                        print(Fore.RED + text, end='')
                        print(Style.RESET_ALL)  # Para restablecer el color a su valor predeterminado
                    else:
                        porc.append(confidence)
                        aux = aux+1
                        # print(text)


                    # cv2.rectangle(frame, (pos_x, pos_y), (x1_, y1_), (0, 0, 0), 4)
                    # cv2.putText(frame, predicted_character, (pos_x, pos_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            else:
                print('No se detectó una mano en : {}'.format(os.path.join(DATA_DIR, dir_, img_path)))
                # Copiar el archivo de la primera imagen que pasa el if correctamente
                first_valid_img = sorted(os.listdir(os.path.join(DATA_DIR, dir_)))[0]
                src_path = os.path.join(DATA_DIR, dir_, first_valid_img)
                dst_path = os.path.join(DATA_DIR, 'no_hand', first_valid_img)
                shutil.copy(src_path, dst_path)

            count += 1
            # Mostrar la imagen
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
        print('El entrenamoiento de {} \t: {}/{} \t{}%'.format(labels_dict[processed_images], aux, count,round((aux/count)*100,2)))
        if len(porc) > 0:
            print('Min {} \t- Max {} \t- Prom {}\t'.format(min(porc),max(porc),round(sum(porc)/len(porc),2)))
        processed_images += 1
        print()

    # Liberar recursos
    hands.close()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Se produjo una excepción: {e}")
    pass