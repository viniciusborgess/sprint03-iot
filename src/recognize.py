#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

def draw_label(img, text, x, y):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, h + 10)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 4, y + baseline - 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def make_trackbar_window(init_scale=110, init_neighbors=5, init_min_size=60, init_thr=80):
    cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controles", 420, 140)
    cv2.createTrackbar("scale x100", "Controles", init_scale, 200, lambda v: None)
    cv2.createTrackbar("neighbors", "Controles", init_neighbors, 15, lambda v: None)
    cv2.createTrackbar("minSize", "Controles", init_min_size, 300, lambda v: None)
    cv2.createTrackbar("lbph_thr", "Controles", init_thr, 200, lambda v: None)

def main():
    ap = argparse.ArgumentParser(description="Detecção e reconhecimento facial em tempo real")
    ap.add_argument("--model", default="data/model/lbph_model.yaml", help="Caminho do modelo LBPH")
    ap.add_argument("--labels", default="data/model/labels.json", help="Caminho do mapeamento label->nome")
    ap.add_argument("--camera", type=int, default=0, help="Índice da webcam")
    args = ap.parse_args()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    if face_cascade.empty():
        raise SystemExit("Falha ao carregar Haar Cascade de face.")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        raise SystemExit("cv2.face não encontrado. Instale 'opencv-contrib-python'.")

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if not model_path.exists() or not labels_path.exists():
        raise SystemExit("Modelo ou labels não encontrados. Treine antes com train_lbph.py.")

    recognizer.read(str(model_path))
    with open(labels_path, "r", encoding="utf-8") as f:
        id_to_name = json.load(f)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a webcam.")

    make_trackbar_window()
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # ler sliders
        scale = max(cv2.getTrackbarPos("scale x100", "Controles"), 110) / 100.0
        neigh = max(cv2.getTrackbarPos("neighbors", "Controles"), 1)
        min_size = max(cv2.getTrackbarPos("minSize", "Controles"), 20)
        thr = max(cv2.getTrackbarPos("lbph_thr", "Controles"), 30)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neigh, minSize=(min_size, min_size)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.10, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes[:2]:
                cx, cy = x + ex + ew//2, y + ey + eh//2
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 2)

            face200 = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_AREA)
            try:
                label_id, distance = recognizer.predict(face200)
            except cv2.error:
                label_id, distance = -1, 9999.0

            name = id_to_name.get(str(label_id), "Desconhecido")
            if distance > thr:
                name = "Desconhecido"

            draw_label(frame, f"{name}  dist={distance:.1f}", x, y - 10)

        fps = 1.0 / (time.time() - last) if time.time() != last else 0.0
        last = time.time()
        draw_label(frame, f"scale={scale:.2f} neigh={neigh} minSize={min_size}px thr={thr}", 10, 25)
        draw_label(frame, f"FPS: {fps:.1f}", 10, 50)

        cv2.imshow("Reconhecimento - LBPH", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
