import cv2
import sys
import os
import datetime
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[4]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()

print(tracker)
# load video
video = cv2.VideoCapture('src/1.mp4')

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
video_codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_output = cv2.VideoWriter('src/out.mp4', video_codec, fps, (frame_width, frame_height))

if not video.isOpened():
    print('[ERROR] video file not loaded')
    sys.exit()
# capture first frame
ok, frame = video.read()
if not ok:
    print('[ERROR] no frame captured')
    sys.exit()
print('[INFO] video loaded and frame capture started')
bbox = cv2.selectROI(frame)
print('[INFO] select ROI and press ENTER or SPACE')
print('[INFO] cancel selection by pressing C')
print(bbox)
ok = tracker.init(frame, bbox)
if not ok:
    print('[ERROR] tracker not initialized')
    sys.exit()
print('[INFO] tracker was initialized on ROI')
# random generate a colour for bounding box
colours = (randint(0, 255), randint(0, 255), randint(0, 255))
# loop through all frames of video file
i = -1
while True:
    ok, frame = video.read()
    if not ok:
        print('[INFO] end of video file reached')
        break
    # update position of ROI based on tracker prediction
    ok, bbox = tracker.update(frame)
    # test print coordinates of predicted bounding box for all frames
    print(ok, bbox)
    if ok == True:
        i += 1
        (x, y, w, h) = [int(v) for v in bbox]
        # use predicted bounding box coordinates to draw a rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), colours, 3)
        with open(f'output/{i}.txt', 'w') as f:
            to_write = f'0 {((x+w)/2)/frame_width} {((y+h)/2)/frame_height} {w/frame_width} {h/frame_height}'
            f.writelines(to_write)
            cv2.imwrite(f'output/{i}.jpg', frame)
        cv2.putText(frame, str(tracker_type), (10, 30), cv2.QT_FONT_NORMAL, 1, (255, 255, 255))

    else:
        # if prediction failed and no bounding box coordinates are available
        cv2.putText(frame, 'No Track', (10, 30), cv2.QT_FONT_NORMAL, 1, (0, 0, 255))

    # display object track
    cv2.imshow('Single Track', frame)
    video_output.write(frame)
    # press 'q' to break loop and close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
video_output.release()
cv2.destroyAllWindows()