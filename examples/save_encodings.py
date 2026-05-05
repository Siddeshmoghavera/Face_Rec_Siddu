import face_recognition
import os
import pickle

train_folder = "../tests/test_images/train"

known_encodings = []
known_names = []

print("Encoding faces...")

for file in os.listdir(train_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(train_folder, file)

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
            print(f"✓ Encoded {file}")

# Save to file
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✅ Encodings saved to encodings.pkl")