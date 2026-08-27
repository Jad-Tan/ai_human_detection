import cv2
from ultralytics import YOLO
import pandas as pd

# 1. Load YOLO model
model = YOLO("yolov8m.pt")

# 2. Open input CCTV video
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Read video properties for VideoWriter initialization
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 4. Initialize VideoWriter ('mp4v' codec writes standard .mp4 files)
output_path = "output_sample.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

person_trajectories = {}
frame_idx = 0

print(f"Processing {total_frames} frames to file... Please wait.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_idx += 1

    # Run native tracking with low confidence & high resolution settings
    results = model.track(
        source=frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],
        conf=0.12,
        imgsz=1280,
        verbose=False
    )

    # Log trajectories for all re-identified IDs
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_x = round(float((x1 + x2) / 2.0), 1)
            center_y = round(float((y1 + y2) / 2.0), 1)

            if track_id not in person_trajectories:
                person_trajectories[track_id] = []

            person_trajectories[track_id].append({
                "frame": frame_idx,
                "x": center_x,
                "y": center_y
            })

    # Render bounding boxes onto frame
    annotated_frame = results[0].plot()

    # Write processed frame directly to video file
    out.write(annotated_frame)

    # Print terminal progress updates every 100 frames
    if frame_idx % 100 == 0:
        print(f"Progress: {frame_idx} / {total_frames} frames written.")

# 5. Release video streams and save trajectory log
cap.release()
out.release()
cv2.destroyAllWindows()

# 1. Set video frame rate to convert frame numbers to timestamps
video_fps = fps if 'fps' in locals() else 30.0

# 2. Flatten trajectory dictionary into row entries
excel_rows = []

for track_id, trajectory in person_trajectories.items():
    for log in trajectory:
        excel_rows.append({
            "Track ID": int(track_id),
            "Frame Number": int(log["frame"]),
            "Timestamp (s)": round(log["frame"] / video_fps, 2),
            "Center X (px)": log["x"],
            "Center Y (px)": log["y"]
        })

# 3. Create pandas DataFrame
df_detections = pd.DataFrame(excel_rows)

# 4. Sort data sequentially by Frame Number and Track ID
df_detections = df_detections.sort_values(by=["Frame Number", "Track ID"]).reset_index(drop=True)

# 5. Build a Person Summary aggregation table
df_summary = df_detections.groupby("Track ID").agg(
    First_Frame=("Frame Number", "min"),
    Last_Frame=("Frame Number", "max"),
    Total_Frames_Visible=("Frame Number", "count"),
    Start_X=("Center X (px)", "first"),
    Start_Y=("Center Y (px)", "first"),
    End_X=("Center X (px)", "last"),
    End_Y=("Center Y (px)", "last")
).reset_index()

# 6. Save directly to Excel with multiple sheets
with pd.ExcelWriter("cctv_person_tracking_log.xlsx", engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary & Analytics", index=False)
    df_detections.to_excel(writer, sheet_name="Detailed Detections", index=False)

# 7. Also save a plain CSV file for raw data import
df_detections.to_csv("cctv_person_tracking_log.csv", index=False)

print("Spreadsheets saved:")
print(" - cctv_person_tracking_log.xlsx (Excel with Summary + Raw tabs)")
print(" - cctv_person_tracking_log.csv (Raw comma-separated data)")