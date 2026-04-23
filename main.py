import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Создаем конфигурацию для FaceLandmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# Создаем детектор
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (680, 540))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Конвертируем numpy-изображение в формат mediapipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        # Получаем координаты точек губ (индексы могут отличаться)
        top_lip = face_landmarks[13]
        bottom_lip = face_landmarks[14]

        h, w, _ = img.shape
        x_top, y_top = int(top_lip.x * w), int(top_lip.y * h)
        x_bot, y_bot = int(bottom_lip.x * w), int(bottom_lip.y * h)

        # Рисуем точки на изображении
        cv2.circle(img, (x_top, y_top), 2, (0, 255, 0), -1)
        cv2.circle(img, (x_bot, y_bot), 2, (0, 255, 0), -1)

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()