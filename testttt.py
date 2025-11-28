import cv2
from pathlib import Path
import torch
import numpy as np

# Model dosya yolun
model_path = Path(r"C:\Users\karat\Desktop\New_folder_(3)\Mantar_Tespiti\yolov5\runs\train\mantar_modeli7\weights\best.pt")

# Cihaz seçimi: GPU varsa 'cuda', yoksa 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Modeli yükle
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)

# Kamerayı aç
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

print("Canlı görüntü başladı. Fotoğraf çekmek için 'a', çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı")
        break

    # Canlı görüntüyü göster
    cv2.imshow('Kamera', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        # Fotoğrafı modele gönder
        results = model(frame)

        # Sonuçları çiz
        result_img = np.squeeze(results.render())

        # Sonucu ayrı pencerede göster
        cv2.imshow("Mantar Tespiti", result_img)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
