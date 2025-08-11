import sys, os, json, csv
from tqdm import tqdm
import cv2
import mediapipe as mp
import numpy as np

def extract(video_path, out_prefix="out"):
    # Resolve to absolute paths
    video_path = os.path.abspath(video_path)

    # Define output folder
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    annotated_path = os.path.join(output_folder, f"{out_prefix}_annotated.mp4")
    writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    frames_data = []
    csv_path = os.path.join(output_folder, f"{out_prefix}_landmarks.csv")
    json_path = os.path.join(output_folder, f"{out_prefix}_landmarks.json")

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=(total_frames if total_frames>0 else None), desc="Frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        frame_record = {
            "frame": frame_idx,
            "time": frame_idx / fps,
            "width": width,
            "height": height,
            "landmarks": None
        }

        if results.pose_landmarks:
            lm_list = []
            for lm in results.pose_landmarks.landmark:
                lm_list.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 0.0))
                })
            frame_record["landmarks"] = lm_list
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frames_data.append(frame_record)
        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    pose.close()

    with open(json_path, "w") as f:
        json.dump({
            "source_video": os.path.basename(video_path),
            "fps": fps,
            "frames": frames_data
        }, f)

    with open(csv_path, "w", newline='') as cf:
        writer_csv = csv.writer(cf)
        header = ["frame", "time"]
        num_landmarks = 33
        for i in range(num_landmarks):
            header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z", f"lm{i}_vis"]
        writer_csv.writerow(header)

        for fr in frames_data:
            row = [fr["frame"], fr["time"]]
            if fr["landmarks"] is None:
                row += ["" for _ in range(num_landmarks*4)]
            else:
                for lm in fr["landmarks"]:
                    row += [lm["x"], lm["y"], lm["z"], lm["visibility"]]
            writer_csv.writerow(row)

    print("Saved:", json_path, csv_path, annotated_path)
    return json_path, csv_path, annotated_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pose.py path/to/input_video.mp4 [out_prefix]")
        sys.exit(1)
    vp = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "out"
    extract(vp, prefix)

