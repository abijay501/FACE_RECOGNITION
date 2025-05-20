import cv2
import numpy as np
import os

# Paths
dataset_path = "dataset"
model_save_path = "face_recognizer.yml"

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 1: Prepare training data
def prepare_training_data(data_folder_path):
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
        print(f"[INFO] Created '{data_folder_path}' directory.")
        print("[ERROR] Please add at least one subfolder with face images before training.")
        return [], [], {}

    dirs = os.listdir(data_folder_path)
    if not dirs:
        print("[ERROR] 'dataset' directory is empty. Add at least one folder with images.")
        return [], [], {}

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for dir_name in dirs:
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        if not os.path.isdir(subject_dir_path):
            continue

        label_map[current_label] = dir_name
        image_names = os.listdir(subject_dir_path)

        for image_name in image_names:
            img_path = os.path.join(subject_dir_path, image_name)
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARNING] Skipped unreadable image: {img_path}")
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                if len(faces_rects) == 0:
                    print(f"[WARNING] No face detected in: {img_path}")
                    continue

                for (x, y, w, h) in faces_rects:
                    face = gray[y:y+h, x:x+w]
                    faces.append(face)
                    labels.append(current_label)
            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
        current_label += 1

    return faces, labels, label_map

# Step 2: Train recognizer
def train_model():
    print("[INFO] Preparing training data...")
    faces, labels, label_map = prepare_training_data(dataset_path)

    if len(faces) == 0 or len(labels) == 0:
        raise ValueError("No training data found. Please add face images to 'dataset'.")

    print(f"[INFO] Training with {len(faces)} face(s)...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_save_path)
    print("[INFO] Model trained and saved.")
    return label_map

# Step 3: Recognize faces in real time
def recognize_faces(label_map):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_save_path)

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting face recognition. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)
            label_text = f"{label_map.get(label, 'Unknown')} ({int(confidence)})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    try:
        if not os.path.exists(model_save_path):
            label_map = train_model()
        else:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError("Missing 'dataset' directory.")
            label_map = {i: name for i, name in enumerate(os.listdir(dataset_path))}
        recognize_faces(label_map)
    except Exception as e:
        print("Error:", e)