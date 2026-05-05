import face_recognition
import cv2
import os
import pickle
from datetime import datetime

STRONG_MATCH_DISTANCE = 0.50
MANUAL_CHECK_MIN_CONFIDENCE = 47.0
MANUAL_CHECK_MAX_CONFIDENCE = 55.0

# Load known faces from train folder
print("Loading saved encodings...")

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

# Convert to your current format
known_encodings = {}
known_names = data["names"]

for name, encoding in zip(data["names"], data["encodings"]):
    known_encodings[name] = encoding

print(f"\nTotal known people: {len(known_names)}")
print("-" * 50)

# Video input
video_file = input("\nEnter the video filename (in test folder): ")

full_path = f"../tests/test_images/test/{video_file}"

try:
    video_capture = cv2.VideoCapture(full_path)

    if not video_capture.isOpened():
        print(f"✗ Cannot open video file: {full_path}")
        exit()

    print("\n✓ Video opened successfully")

    # Video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 25
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}, Frames: {frame_count}")
    print(f"Resolution: {width}x{height}")

    # Clean filename
    base_name = os.path.splitext(os.path.basename(video_file))[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "../tests/test_images/test/output"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"output_with_boxes_{base_name}_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # ✅ FIXED: Correct codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("✗ Failed to create video writer!")
        exit()

    frame_num = 0
    process_interval = max(1, fps // 2)
    detected_people = {}

    process_this_frame = True
    face_locations = []
    face_names = []

    print(f"\nProcessing every {process_interval} frame(s)...")
    print(f"Output: {output_path}")
    print("-" * 50)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_num += 1
        frame_to_save = frame.copy()

        # Process every other frame (toggle)
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(
                    list(known_encodings.values()),
                    face_encoding
                )
                sorted_indices = distances.argsort()
                best_idx = sorted_indices[0]
                second_idx = sorted_indices[1]
                best_distance = distances[best_idx]
                second_distance = distances[second_idx]
                best_name = known_names[best_idx]
                margin = second_distance - best_distance
                confidence = (1 - best_distance) * 100

                if MANUAL_CHECK_MIN_CONFIDENCE <= confidence <= MANUAL_CHECK_MAX_CONFIDENCE:
                    print(
                        f"⚠️ Weak / uncertain zone at frame {frame_num}: {best_name} "
                        f"({confidence:.1f}% confidence, distance: {best_distance:.4f})"
                    )
                    manual_choice = input(
                        f"Manual check for frame {frame_num}. Is it {best_name}? "
                        "[y = same / n = different / u = not sure]: "
                    ).strip().lower()

                    if manual_choice == "y":
                        name = best_name
                        detected_people[name] = detected_people.get(name, 0) + 1
                        print(f"✓ Manual check accepted as {best_name}")
                    elif manual_choice == "u":
                        name = "NOT_SURE"
                        detected_people[name] = detected_people.get(name, 0) + 1
                        print(f"? Marked as not sure for {best_name}")
                    else:
                        name = "UNKNOWN"
                        detected_people[name] = detected_people.get(name, 0) + 1
                        print(f"✗ Manual check rejected {best_name}; marked as UNKNOWN")
                elif best_distance < STRONG_MATCH_DISTANCE and margin > 0.05:
                    name = best_name
                    detected_people[name] = detected_people.get(name, 0) + 1
                else:
                    name = "UNKNOWN"
                    detected_people["UNKNOWN"] = detected_people.get("UNKNOWN", 0) + 1
                face_names.append(name)

        # toggle processing flag
        process_this_frame = (frame_num % 3 == 0)

        # Scale coordinates back to original frame size
        scaled_locations = [
            (int(top*4), int(right*4), int(bottom*4), int(left*4))
            for top, right, bottom, left in face_locations
        ]

        # Draw boxes and labels
        for (top, right, bottom, left), name in zip(scaled_locations, face_names):
            color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
            label = name

            cv2.rectangle(frame_to_save, (left, top), (right, bottom), color, 3)

            label_y = top - 15 if top > 30 else bottom + 30
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            cv2.rectangle(
                frame_to_save,
                (left, label_y - label_size[1] - 8),
                (left + label_size[0] + 8, label_y + baseline + 4),
                color,
                cv2.FILLED
            )
            cv2.putText(
                frame_to_save,
                label,
                (left + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Write frame to output
        out.write(frame_to_save)

        # Print progress
        if frame_num % 30 == 0:
            print(f"  Processing frame {frame_num}/{frame_count}...")

        # ✅ OPTIONAL: Live preview (press ESC to exit)
        cv2.imshow("Processing - Press ESC to exit", frame_to_save)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(output_path):
        print(f"\n✅ Saved: {output_path}")
        print(f"Size: {os.path.getsize(output_path)/(1024*1024):.2f} MB")

    print("\n===== SUMMARY =====")
    for person, count in sorted(detected_people.items(), key=lambda x: x[1], reverse=True):
        print(f"{person}: {count} times")
    
    print("\n===== FINAL RESULT =====")

    if detected_people:
        final_people = []

        for person, count in detected_people.items():
            if person != "UNKNOWN" and count >= 1:
                final_people.append(person)

        if detected_people.get("UNKNOWN", 0) >= 1:
            final_people.append("UNKNOWN")

        if final_people:
            print("Persons found:", ", ".join(final_people))
        else:
            print("No reliable detections")

    else:
        print("No faces detected")
except Exception as e:
    print(f"✗ Error: {e}")