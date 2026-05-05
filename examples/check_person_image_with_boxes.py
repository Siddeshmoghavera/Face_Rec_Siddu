import face_recognition
import cv2
import os
import pickle
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


STRONG_MATCH_DISTANCE = 0.50
MANUAL_CHECK_MIN_CONFIDENCE = 47.0
MANUAL_CHECK_MAX_CONFIDENCE = 55.0

# Load known faces from train folder
import pickle

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

# Image file input
image_file = input("\nEnter the image filename (in test folder): ")
full_path = f"../tests/test_images/test/{image_file}"

try:
    if not os.path.exists(full_path):
        print(f"✗ Image file not found: {full_path}")
        exit()
    
    # Load image
    unknown_image = face_recognition.load_image_file(full_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    unknown_locations = face_recognition.face_locations(unknown_image)
    
    print(f"\n✓ Image loaded successfully")
    print(f"Image size: {unknown_image.shape}")
    print("-" * 50)
    
    if not unknown_encodings:
        print("✗ No faces found in the image!")
        exit()
    
    print(f"Found {len(unknown_encodings)} face(s)")
    print("-" * 50)
    
    # Open image with PIL for drawing
    pil_image = Image.open(full_path)
    draw = ImageDraw.Draw(pil_image)
    
    detected_people = {}
    
    for (top, right, bottom, left), face_encoding in zip(unknown_locations, unknown_encodings):
        # Compare against known faces
        results = face_recognition.compare_faces(
            list(known_encodings.values()),
            face_encoding,
            tolerance=0.6
        )
        
        distances = face_recognition.face_distance(
            list(known_encodings.values()),
            face_encoding
        )
        
        best_match_idx = distances.argmin()
        best_match_name = known_names[best_match_idx]
        best_match_distance = distances[best_match_idx]
        confidence = (1 - best_match_distance) * 100
        
        if MANUAL_CHECK_MIN_CONFIDENCE <= confidence <= MANUAL_CHECK_MAX_CONFIDENCE:
            print(
                f"⚠️ Weak / uncertain zone: {best_match_name} "
                f"({confidence:.1f}% confidence, distance: {best_match_distance:.4f})"
            )
            manual_choice = input(
                f"Manual check for this face. Is it {best_match_name}? "
                "[y = same / n = different / u = not sure]: "
            ).strip().lower()

            if manual_choice == "y":
                detected_people[best_match_name] = detected_people.get(best_match_name, 0) + 1
                color = "orange"
                label = f"{best_match_name}\nManual Check\n({confidence:.1f}%)"
                print(f"✓ Manual check accepted as {best_match_name}")
            elif manual_choice == "u":
                detected_people["NOT_SURE"] = detected_people.get("NOT_SURE", 0) + 1
                color = "yellow"
                label = f"NOT SURE\n({confidence:.1f}%)"
                print(f"? Marked as not sure for {best_match_name}")
            else:
                color = "red"
                label = f"UNKNOWN\n({confidence:.1f}%)"
                detected_people["UNKNOWN"] = detected_people.get("UNKNOWN", 0) + 1
                print(f"✗ Manual check rejected {best_match_name}; marked as UNKNOWN")
        elif best_match_distance < STRONG_MATCH_DISTANCE:
            detected_people[best_match_name] = detected_people.get(best_match_name, 0) + 1
            color = "green"
            label = f"{best_match_name}\n({confidence:.1f}%)"
            print(f"✓ Found {best_match_name} (confidence: {confidence:.1f}%)")
        else:
            color = "red"
            label = f"UNKNOWN\n({confidence:.1f}%)"
            if "UNKNOWN" not in detected_people:
                detected_people["UNKNOWN"] = 0
            detected_people["UNKNOWN"] += 1
            print(f"✗ Unknown person (best match: {best_match_name} @ {best_match_distance:.4f})")
        
        # Draw rectangle
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)
        
        # Draw label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        label_y = top - 30 if top > 30 else bottom + 10
        draw.text((left + 5, label_y), label, fill=color, font=font)
    
    # Save output image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_with_boxes_{os.path.splitext(image_file)[0]}_{timestamp}.jpg"
    output_path = f"../tests/test_images/test/{output_filename}"
    
    pil_image.save(output_path)
    
    print("\n" + "=" * 50)
    print("IMAGE ANALYSIS SUMMARY")
    print("=" * 50)
    
    for person, count in sorted(detected_people.items(), key=lambda x: x[1], reverse=True):
        print(f"{person}: Detected {count} time(s)")
    
    print(f"\n✓ Output image saved: {output_path}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
