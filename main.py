import os
import zipfile
import shutil
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

app = FastAPI()
uploaded_directory = "uploaded_files"
image_directory = "images_folder"
processed_images = set()

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

@app.post("/process_images/")
async def process_images(zip_file: UploadFile):
    # Create a temporary directory to store the extracted images
    os.makedirs(uploaded_directory, exist_ok=True)
    os.makedirs(image_directory, exist_ok=True)

    # Save the uploaded zip file
    zip_file_path = os.path.join(uploaded_directory, zip_file.filename)
    with open(zip_file_path, "wb") as f:
        f.write(zip_file.file.read())

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(image_directory)
    result_group = []
    result_solo=[]
    for root, _, files in os.walk(image_directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)

                # Check if the image has already been processed
                if image_path in processed_images:
                    continue

                image = Image.open(image_path)

                # Transform the image to a tensor
                image = F.to_tensor(image).unsqueeze(0)

                # Make predictions using the model
                with torch.no_grad():
                    prediction = model(image)

                # Extract bounding boxes and labels
                boxes = prediction[0]['boxes']
                labels = prediction[0]['labels']

                # Set a confidence threshold
                confidence_threshold = 0.7

                # Filter out low-confidence detections
                high_confidence_boxes = [box for i, box in enumerate(boxes) if labels[i] == 1 and prediction[0]['scores'][i] > confidence_threshold]
                num_persons = len(high_confidence_boxes)

                if num_persons > 1:
                    result_group.append(filename)
                    # Move the image to the "GroupImage" directory
                    group_image_dir = "GroupImage"
                    os.makedirs(group_image_dir, exist_ok=True)
                    real_path = os.path.join(group_image_dir, filename)
                    shutil.move(image_path, real_path)
                    processed_images.add(image_path)  # Add to processed set
                else:
                    result_solo.append(filename)
                    # Move the image to the "SoloImage" directory
                    solo_image_dir = "SoloImage"
                    os.makedirs(solo_image_dir, exist_ok=True)
                    real_path = os.path.join(solo_image_dir, filename)
                    shutil.move(image_path, real_path)
                    processed_images.add(image_path)  # Add to processed set

    return {'result_groupImages': result_group,
            'result_soloImage': result_solo}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
