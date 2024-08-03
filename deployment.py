import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# Load the trained model
model = YOLO(r"C:\Users\shubham lokare\Downloads\best.pt")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None

    # Get video writer initialized to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding box coordinates
            confidences = result.boxes.conf  # Get confidence scores
            class_ids = result.boxes.cls  # Get class indices

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)  # Extract coordinates directly from box
                label = model.names[int(class_id)] if hasattr(model, 'names') else 'unknown'  # Assuming model.names exists
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame into the output video
        out.write(frame)

    cap.release()
    out.release()
    return temp_output.name

st.title("Vials counting App")
st.write("Upload a video and see object detection in action!")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    temp_file_path = tfile.name

    with st.spinner('Processing video...'):
        output_video_path = process_video(temp_file_path)

    if output_video_path:
        st.success('Video processing complete!')
        st.video(output_video_path)
    else:
        st.error('Error processing video.')
