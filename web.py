import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# Load a pretrained YOLOv8 model
model = YOLO("MODELS/best.pt")
obb = YOLO("MODELS/yolov8s.pt")

# Define camera parameters
focal_length = 700  # Replace with your camera's actual focal length in pixels
object_width = 50  # Replace with the actual width of the object in cm

def get_dist(rectangle_params, image):
    # Extract the bounding box dimensions
    x1, y1, x2, y2 = rectangle_params
    width = x2 - x1

    # Calculate distance
    dist = (object_width * focal_length) / width

    # Define text parameters
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)

    # Write distance on the image
    image = cv2.putText(image, f'Distance from Camera in CM: {int(dist)}', org, font, 
                       fontScale, color, 2, cv2.LINE_AA)

    return image

def image(source, confidence):
    img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    results = model(img, conf=confidence)
    annotated_frame = results[0].plot()
    resized_frame = cv2.resize(annotated_frame, (500, 500))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    return img, rgb_frame

def video(source, confidence):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    with open(temp_file.name, 'wb') as f:
        f.write(source)

    cap = cv2.VideoCapture(temp_file.name)
    desired_width = 640
    desired_height = 480
    frame_skip = 2
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Display video frames with Streamlit
    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (desired_width, desired_height))

            # Get results from both models
            results = model(frame, conf=confidence)
            res_obb = obb(frame, conf=confidence)

            # Annotate frame with detection results
            annotated_frame = results[0].plot()

            # Check for cars and calculate distance using the second model
            for result in res_obb[0].boxes:
                class_index = int(result.cls.item())
                if class_index < len(obb.names):
                    class_name = obb.names[class_index]
                    if class_name == 'car':
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = result.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Width in pixels
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height

                        # Define the threshold area for "too close" detection
                        close_threshold_area = 10000
                        if area > close_threshold_area:
                            print("TOO CLOSE!!")
                            annotated_frame = get_dist([x1, y1, x2, y2], annotated_frame)
                            cv2.putText(annotated_frame, "Car Too Close!", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display frame using Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)
        else:
            break

    cap.release()
    os.remove(temp_file.name)

def live_feed(confidence):
    st.header("Live Camera Feed")

    # Initialize the video capture from the default camera (usually camera index 0)
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Failed to capture image")
            break

        # Process the frame with YOLO model
        results = model(frame, conf=confidence)
        annotated_frame = results[0].plot()

        if annotated_frame is not None:
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)

    cap.release()
    stframe = st.empty()

def main():
    st.session_state.camera_active = False
    
    st.title("YOLOv8 Object Detection")

    # Sidebar for user inputs
    st.sidebar.markdown("""
    <h2 style='font-weight: bold; font-size: 30px;'>Settings</h2>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.05)

    st.sidebar.markdown("---")

    # File upload section
    st.sidebar.header("Upload a file")
    uploaded_file = st.sidebar.file_uploader("Choose an image or video", type=["png", "jpg", "jpeg", "mp4"])
    st.sidebar.markdown("---")

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]

        if file_type == 'image':
            original_image, processed_image = image(uploaded_file.read(), confidence)
            st.sidebar.write("INPUT")
            st.sidebar.image(original_image, channels="BGR")

            # st.header("Processed Image")
            st.image(processed_image, channels="RGB")

        elif file_type == 'video':
            st.sidebar.video(uploaded_file)
            # st.header("Processed Video")
            video(uploaded_file.read(), confidence)

    
    # Button to start live camera feed
    if st.sidebar.button("Start Camera Feed"):
        st.session_state.camera_active = True

        if st.sidebar.button("Stop Camera Feed"):
            st.session_state.camera_active = False
    

    if st.session_state.camera_active == True:
        live_feed(confidence)

if __name__ == "__main__":
    main()
