import cv2
import os
import numpy as np
from ultralytics import YOLO

detector = YOLO('yolov8n.pt') 

DATA_ROOT = r'C:\Users\User\OneDrive\Desktop\important\datacleaning\data'
VIDEO_DIR = os.path.join(DATA_ROOT, 'Videos')
CLEANED_DIR = r'C:\Users\User\OneDrive\Desktop\important\datacleaning\cleaned_dataset'
os.makedirs(CLEANED_DIR, exist_ok=True)

track_history = {}

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.MOV')]

for vid_file in video_files:
    video_id = vid_file.replace('video', '').replace('.MOV', '')
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, vid_file))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        results = detector.track(frame, persist=True, device=0, verbose=False) 
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                is_dangerous = False
                if track_id in track_history:
                    prev_cx, prev_cy = track_history[track_id][-1]
                    movement = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    
                    if movement > 10:
                        is_dangerous = True
                
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))
                
                color = (0, 0, 255) if is_dangerous else (0, 255, 0)
                label = "DANGER: CROSSING" if is_dangerous else "SAFE: NOT CROSSING"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_dangerous:
                    save_name = f"live_danger_vid{video_id}_f{frame_id}.jpg"
                    cv2.imwrite(os.path.join(CLEANED_DIR, save_name), frame)

        cv2.imshow("Pedestrian Intent Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()