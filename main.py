from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st
from sort.sort import *
import util
from util import get_car, read_license_plate, write_csv
import tempfile
import os
import pandas as pd
from datetime import datetime
import csv
from scipy.interpolate import interp1d
import ast
import time

st.set_page_config(
    page_title="ANPR Vehicle Tracker",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #764ba2;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
    }
    
    .dataframe {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .modal-video {
        position: fixed;
        top: 5%;
        left: 5%;
        width: 90%;
        height: 90%;
        background-color: rgba(0,0,0,0.95);
        z-index: 9999;
        padding: 20px;
        border-radius: 15px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .modal-video video {
        width: 100%;
        height: auto;
        max-height: 85vh;
        border-radius: 10px;
    }
    
    .modal-video button {
        background-color: #ff4b4b;
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 15px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .modal-video button:hover {
        background-color: #ff3333;
        transform: scale(1.05);
    }
    
    .vehicle-counter {
        text-align: center;
        padding: 1rem;
    }
    
    .vehicle-counter-title {
        color: #FF6B35;
        font-size: 1.44rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .vehicle-counter-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    
    .centered-video {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 0;
    }
    
    .centered-video img {
        width: 80% !important;
        height: 80% !important;
        max-width: 80% !important;
        object-fit: contain;
    }
    
    .completion-message {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    .completion-message h2 {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .completion-message p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Smart Traffic Management with ANPR and ATCC")

st.subheader("Upload Video for Analysis")

col_upload, col_analyze = st.columns([0.8, 0.2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
        label_visibility="collapsed"
    )

if 'previous_file_id' not in st.session_state:
    st.session_state.previous_file_id = None

if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False

current_file_id = id(uploaded_file) if uploaded_file is not None else None

if current_file_id != st.session_state.previous_file_id:
    st.session_state.previous_file_id = current_file_id
    st.session_state.analyze_clicked = False
    st.session_state.analysis_completed = False

with col_analyze:
    if uploaded_file is not None:
        if st.button("Analyze Video for ANPR and ATCC", key="analyze_button", width='stretch'):
            st.session_state.analyze_clicked = True
            st.session_state.analysis_completed = False

if uploaded_file is None:
    st.stop()

if not st.session_state.analyze_clicked:
    st.info("Click 'Analyze Video for ANPR and ATCC' to start processing")
    st.stop()

if st.session_state.analysis_completed:
    progress_placeholder = st.empty()
    progress_placeholder.markdown("""
    <div class="completion-message">
        <h2>✓ Video Analysis Completed</h2>
        <p>All vehicles have been processed successfully</p>
    </div>
    """, unsafe_allow_html=True)
    
    frame_placeholder = st.empty()
    vehicle_counter_placeholder = st.empty()
    results_table_placeholder = st.empty()
    download_button_placeholder = st.empty()
    
    if 'final_results_df' in st.session_state and st.session_state.final_results_df is not None:
        results_table_placeholder.dataframe(
            st.session_state.final_results_df,
            width='stretch',
            hide_index=True
        )
        
        csv_bytes = st.session_state.final_results_df.to_csv(index=False).encode('utf-8')
        download_button_placeholder.download_button(
            label="Download Results CSV",
            data=csv_bytes,
            file_name=f'vehicle_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            key=f"download_results_final"
        )
    
    st.stop()

tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())
media_path = tfile.name

@st.cache_resource
def load_models():
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    return coco_model, license_plate_detector

try:
    coco_model, license_plate_detector = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

mot_tracker = Sort()
vehicles = [2, 3, 5, 7]
vehicle_type_map = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

cap = cv2.VideoCapture(media_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_placeholder = st.empty()
progress_bar = st.progress(0)
vehicle_counter_placeholder = st.empty()
results_table_placeholder = st.empty()
download_button_placeholder = st.empty()

os.makedirs("./result", exist_ok=True)
output_path = os.path.join("./result", "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

results_csv_path = './results.csv'
pd.DataFrame(columns=[
    'S.No', 'Vehicle Id', 'Vehicle Type', 'License Plate number', 'Confidence Score', 'Time Seen'
]).to_csv(results_csv_path, index=False)

def update_vehicle_counter(detections, vehicle_type_map, placeholder):
    type_counts = {t: 0 for t in vehicle_type_map.values()}
    total = 0
    
    for det in detections.boxes.data.tolist():
        _, _, _, _, _, class_id = det
        cls = vehicle_type_map.get(int(class_id))
        if cls:
            type_counts[cls] += 1
            total += 1

    cols = placeholder.columns(len(type_counts) + 1)
    
    cols[0].markdown(f"""
    <div class="vehicle-counter">
        <div class="vehicle-counter-title">Total Vehicles</div>
        <div class="vehicle-counter-value">{total}</div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, (vtype, count) in enumerate(type_counts.items(), 1):
        cols[i].markdown(f"""
    <div class="vehicle-counter">
        <div class="vehicle-counter-title">{vtype}</div>
        <div class="vehicle-counter-value">{count}</div>
    </div>
    """, unsafe_allow_html=True)

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    
    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_length_x=50, line_length_y=50):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    xi1 = max(ax1, bx1)
    yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2)
    yi2 = min(ay2, by2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def flush_results_csv_and_ui(best_vehicle_log, csv_path, frame_key):
    rows = []
    serial_no = 1
    
    for car_id, v in best_vehicle_log.items():
        if v.get('license_plate_text') and v.get('license_plate_text') != 'N/A' and v.get('license_plate_text') != '0':
            rows.append({
                'S.No': serial_no,
                'Vehicle Id': int(car_id),
                'Vehicle Type': v.get('vehicle_type', 'Unknown'),
                'License Plate number': v.get('license_plate_text', 'N/A'),
                'Confidence Score': f"{v.get('best_license_plate_text_score', 0.0):.2f}",
                'Time Seen': v.get('time_seen', 'N/A')
            })
            serial_no += 1
    
    if rows:
        df_results = pd.DataFrame(rows)
        df_results.to_csv(csv_path, index=False)
        
        results_table_placeholder.dataframe(
            df_results,
            width='stretch',
            hide_index=True
        )
        
        return df_results
    
    return None

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

def show_video_modal():
    st.session_state.show_modal = True

def close_video_modal():
    st.session_state.show_modal = False

results = {}
frame_nmr = -1
ret = True
start_time = time.time()

best_vehicle_log = {}

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}
    
    detections = coco_model(frame)[0]
    
    update_vehicle_counter(detections, vehicle_type_map, vehicle_counter_placeholder)
    
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    track_ids = mot_tracker.update(np.asarray(detections_)) if len(detections_) > 0 else []

    license_plates = license_plate_detector(frame)[0]
    
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                vehicle_type = "Unknown"
                for det in detections.boxes.data.tolist():
                    dx1, dy1, dx2, dy2, dscore, dclass_id = det
                    if (int(dclass_id) in vehicles and 
                        abs(dx1 - xcar1) < 50 and abs(dy1 - ycar1) < 50):
                        vehicle_type = vehicle_type_map[int(dclass_id)]
                        break
                
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    },
                    'vehicle_type': vehicle_type
                }
                
                if car_id not in best_vehicle_log or license_plate_text_score > best_vehicle_log[car_id].get('best_license_plate_text_score', 0):
                    best_vehicle_log[car_id] = {
                        'license_plate_text': license_plate_text,
                        'best_license_plate_text_score': license_plate_text_score,
                        'time_seen': datetime.now().strftime("%H:%M:%S"),
                        'vehicle_type': vehicle_type
                    }
                
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 3)
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                
                (text_width, text_height), _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
                text_x = int((x1 + x2 - text_width) / 2)
                text_y = int(y1) - 10
                
                cv2.rectangle(frame, (text_x - 4, text_y - text_height - 4),
                            (text_x + text_width + 4, text_y + 4), (255, 255, 255), -1)
                cv2.putText(frame, license_plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                          1.05, (0, 0, 0), 2, cv2.LINE_AA)
                
                label_text = f"ID {int(car_id)} - {vehicle_type}"
                label_font_scale = 0.6
                label_thickness = 1
                (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
                lbl_x = int(xcar1) + 5
                lbl_y = int(ycar1) - 10
                
                cv2.rectangle(frame, (lbl_x - 4, lbl_y - lbl_h - 4), (lbl_x + lbl_w + 4, lbl_y + 4), (255, 255, 255), -1)
                cv2.putText(frame, label_text, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), label_thickness, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 85), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    now = datetime.now()
    cv2.putText(frame, f'Frame: {frame_nmr}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f'Date: {now.strftime("%Y-%m-%d")}', (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f'Time: {now.strftime("%H:%M:%S")}', (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_html = f'<div class="centered-video"><img src="data:image/png;base64,{{0}}" /></div>'
    frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')

    progress_bar.progress(min(frame_nmr / frame_count, 1.0))
    
    flush_results_csv_and_ui(best_vehicle_log, results_csv_path, frame_nmr)

write_csv(results, 'test.csv')

cap.release()

with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

interpolated_data = interpolate_bounding_boxes(data)

header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

results_df = pd.read_csv('test_interpolated.csv')

license_plate_dict = {}

for car_id in np.unique(results_df['car_id']):
    max_score = np.amax(results_df[results_df['car_id'] == car_id]['license_number_score'])
    best_frame_data = results_df[(results_df['car_id'] == car_id) & 
                                (results_df['license_number_score'] == max_score)].iloc[0]
    
    license_plate_number = best_frame_data['license_number']
    
    license_plate_dict[car_id] = {
        'license_plate_number': license_plate_number
    }

cap = cv2.VideoCapture(media_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    detections_frame = coco_model(frame)[0]

    df_frame = results_df[results_df['frame_nmr'] == frame_nmr]

    for row_indx in range(len(df_frame)):
        try:
            car_bbox_str = df_frame.iloc[row_indx]['car_bbox']
            lp_bbox_str = df_frame.iloc[row_indx]['license_plate_bbox']
            
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                car_bbox_str.replace('[ ', '[').replace('   ', ' ')
                          .replace('  ', ' ').replace(' ', ',')
            )
            x1, y1, x2, y2 = ast.literal_eval(
                lp_bbox_str.replace('[ ', '[').replace('   ', ' ')
                         .replace('  ', ' ').replace(' ', ',')
            )

            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)),
                       (0, 255, 0), 3, line_length_x=50, line_length_y=50)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            car_id = df_frame.iloc[row_indx]['car_id']
            plate_text = license_plate_dict.get(car_id, {}).get('license_plate_number', 'N/A')
            
            if plate_text != '0' and plate_text != 'N/A':
                (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
                text_x = int((x1 + x2 - text_width) / 2)
                text_y = int(y1) - 10

                cv2.rectangle(frame, (text_x - 4, text_y - text_height - 4),
                            (text_x + text_width + 4, text_y + 4), (255, 255, 255), -1)
                cv2.putText(frame, plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                          1.05, (0, 0, 0), 2, cv2.LINE_AA)

            car_id_str = df_frame.iloc[row_indx]['car_id']
            label_text = f"ID {car_id_str} - Unknown"

            best_iou = 0
            best_class = None
            car_box = [float(car_x1), float(car_y1), float(car_x2), float(car_y2)]
            
            detections_frame_list = detections_frame.boxes.data.tolist()
            for det in detections_frame_list:
                try:
                    dx1, dy1, dx2, dy2, dscore, dclass = det
                except Exception:
                    continue
                det_box = [dx1, dy1, dx2, dy2]
                this_iou = iou(car_box, det_box)
                if this_iou > best_iou:
                    best_iou = this_iou
                    best_class = int(dclass)

            if best_class is not None:
                class_name = coco_model.model.names.get(best_class, "Unknown")
                label_text = f"ID {car_id_str} - {class_name}"
            else:
                label_text = f"ID {car_id_str} - Unknown"

            label_font_scale = 0.6
            label_thickness = 1
            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
            
            lbl_x = int(car_x1) + 5
            lbl_y = int(car_y1) - 10

            cv2.rectangle(frame, (lbl_x - 4, lbl_y - lbl_h - 4), (lbl_x + lbl_w + 4, lbl_y + 4), (255, 255, 255), -1)
            cv2.putText(frame, label_text, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), label_thickness, cv2.LINE_AA)

        except Exception as e:
            continue
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 85), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    now = datetime.now()
    cv2.putText(frame, f'Frame: {frame_nmr}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f'Date: {now.strftime("%Y-%m-%d")}', (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f'Time: {now.strftime("%H:%M:%S")}', (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    out.write(frame)

cap.release()
out.release()

final_df = flush_results_csv_and_ui(best_vehicle_log, results_csv_path, frame_nmr + 1)

st.session_state.final_results_df = final_df
st.session_state.analysis_completed = True

progress_bar.empty()

progress_placeholder = st.empty()
progress_placeholder.markdown("""
<div class="completion-message">
    <h2>✓ Video Analysis Completed</h2>
    <p>All vehicles have been processed successfully</p>
</div>
""", unsafe_allow_html=True)

if final_df is not None:
    csv_bytes = final_df.to_csv(index=False).encode('utf-8')
    download_button_placeholder.download_button(
        label="Download Results CSV",
        data=csv_bytes,
        file_name=f'vehicle_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
        key=f"download_results_completed"
    )

if st.session_state.show_modal:
    st.markdown("<div class='modal-video'>", unsafe_allow_html=True)
    st.video(output_path)
    st.button("Close Video", on_click=close_video_modal)
    st.markdown("</div>", unsafe_allow_html=True)