import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import serial
import serial.tools.list_ports
import time
import math

# --- Настройка последовательного порта ---
# Автоматический поиск STM (можно указать вручную)
def find_stm_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Поиск по описанию, VID/PID или просто пробуем
        if "STMicroelectronics" in port.description or "USB Serial" in port.description:
            return port.device
    return None

stm_port = find_stm_port()
if stm_port is None:
    # Если не нашли, введите вручную, например '/dev/ttyUSB0'
    stm_port = "/dev/ttyUSB0"  # подставьте свой

try:
    ser = serial.Serial(stm_port, 115200, timeout=0.1)  # скорость 115200, как на STM
    time.sleep(2)  # ждём инициализации
    print(f"Connected to STM on {stm_port}")
except Exception as e:
    print(f"Could not open serial port: {e}")
    ser = None

# --- Настройка MediaPipe Face Landmarker ---
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Параметры для определения открытия рта
MOUTH_OPEN_THRESHOLD = 0.020  # нормализованное расстояние (в метриках 0..1) ~ 2% от высоты лица
# или в пикселях задать после resize: примерно 10-15 пикселей.

last_mouth_open = False

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (680, 540))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    detection_result = detector.detect(mp_image)

    mouth_open = False

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        top_lip = face_landmarks[13]
        bottom_lip = face_landmarks[14]

        # Вычисляем расстояние между точками (евклидово, но проще по Y)
        dy = abs(top_lip.y - bottom_lip.y)
        # Иногда для надёжности можно брать евклидово, но разница по Y достаточна
        # distance = math.sqrt((top_lip.x - bottom_lip.x)**2 + (top_lip.y - bottom_lip.y)**2)
        distance = dy

        # Порог зависит от размера лица, можно использовать относительный: если расстояние > 0.02 (2% от высоты кадра)
        # Умножать на высоту необязательно, так как координаты нормализованы (0-1)
        if distance > MOUTH_OPEN_THRESHOLD:
            mouth_open = True

        # Визуализация точек
        h, w, _ = img.shape
        x_top, y_top = int(top_lip.x * w), int(top_lip.y * h)
        x_bot, y_bot = int(bottom_lip.x * w), int(bottom_lip.y * h)

        # Цвет точек меняется в зависимости от состояния
        color = (0, 0, 255) if mouth_open else (0, 255, 0)  # красный = открыт, зелёный = закрыт
        cv2.circle(img, (x_top, y_top), 3, color, -1)
        cv2.circle(img, (x_bot, y_bot), 3, color, -1)

        # Дополнительно: отображаем расстояние и состояние
        cv2.putText(img, f"Distance: {distance:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        state_text = "MOUTH OPEN" if mouth_open else "mouth closed"
        cv2.putText(img, state_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Отправка сигнала на STM при изменении состояния
    if mouth_open != last_mouth_open:
        if ser and ser.is_open:
            if mouth_open:
                ser.write(b'1')  # включить лампочку
                print("Sent: ON")
            else:
                ser.write(b'0')  # выключить
                print("Sent: OFF")
        last_mouth_open = mouth_open

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()