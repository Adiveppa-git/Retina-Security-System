import os
from PIL import Image

from backend.app.services.iris_recognition_service import recognize_iris

dataset_path = "project_images"

total = 0
correct = 0

print("\n===== TEST RESULTS =====\n")

for file in os.listdir(dataset_path):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    actual_name = os.path.splitext(file)[0]

    try:
        image = Image.open(
            os.path.join(dataset_path, file)
        )

    except Exception:
        print(f"Skipping invalid image: {file}")
        continue

    predicted_name = recognize_iris(image)

    print(
        f"Actual: {actual_name} | Predicted: {predicted_name}"
    )

    total += 1

    if actual_name == predicted_name:
        correct += 1

accuracy = (correct / total) * 100 if total > 0 else 0

print("\n========================")
print(f"Correct Predictions: {correct}")
print(f"Total Images: {total}")
print(f"Accuracy: {accuracy:.2f}%")
print("========================")