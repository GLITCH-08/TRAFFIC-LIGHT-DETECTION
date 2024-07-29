from ultralytics import YOLO
import cv2

# Load the YOLO models
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

def image(source):
    source = "img/" + source
    results = model(source)
    annotated_frame = results[0].plot()
    resized_frame = cv2.resize(annotated_frame, (800, 600))
    cv2.imshow("YOLOv8 Detection", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video(video_path):
    video_path = "video/" + video_path
    cap = cv2.VideoCapture(video_path)
    desired_width = 640
    desired_height = 480
    frame_skip = 2
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (desired_width, desired_height))

            # Get results from both models
            results = model(frame, conf=0.7)
            res_obb = obb(frame, conf=0.7)

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

            cv2.imshow("YOLOv8 Inference", annotated_frame) 

            # Check for the 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

def check():
    model.predict("video/test.mp4", show=True, conf=0.7)

# Call video or image function as needed
# video('test.mp4')
# image('test.jpg')
