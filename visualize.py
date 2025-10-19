import cv2
import pandas as pd
import numpy as np
import ast

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_length_x=50, line_length_y=50):
    # Changed to draw a full rectangle instead of corner lines
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


results = pd.read_csv('test_interpolated.csv')
print("WOW, read from interpolated!!!!!!!!!!!!!!")

video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox']
                                      .iloc[0].replace('[ ', '[').replace('   ', ' ')
                                      .replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
    license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # draw car border (now full rectangle)
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox']
                                                              .replace('[ ', '[').replace('   ', ' ')
                                                              .replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)),
                        (0, 255, 0), 3, line_length_x=50, line_length_y=50)

            # draw smaller license plate border
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox']
                                              .replace('[ ', '[').replace('   ', ' ')
                                              .replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            # new OCR text display style (reduced by ~12%)
            plate_text = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
            (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)

            text_x = int((x1 + x2 - text_width) / 2)
            text_y = int(y1) - 10  # slightly above the license plate box

            # background for text (white box) reduced slightly
            cv2.rectangle(frame, (text_x - 4, text_y - text_height - 4),
                          (text_x + text_width + 4, text_y + 4), (255, 255, 255), -1)

            # put black text on top (slightly smaller font)
            cv2.putText(frame, plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.05, (0, 0, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

out.release()
cap.release()
