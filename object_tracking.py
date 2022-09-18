import cv2
import sys
import os
import datetime
from random import randint

#! in this project we use MedianFlow object tracker to track object bounding box
#! in continuous frames of a video
tracker_type = 'MEDIANFLOW'
colours = (255, 255, 0)
tracker = cv2.legacy.TrackerMedianFlow_create()
video = cv2.VideoCapture('src/1.mp4')

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
video_codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_output = cv2.VideoWriter('src/out.mp4', video_codec, fps, (frame_width, frame_height))

def main():
    #! here we check if video could be opened or not
    if not video.isOpened():
        print('[ERROR] video file not loaded')
        sys.exit()

    #! capture first frame of video for tracker and object bounding box initialization
    ret, frame = video.read()
    if not ret:
        print('[ERROR] no frame captured')
        sys.exit()

    print('[INFO] video loaded and frame capture started')
    #! select a ROI (Region of Interest) in frame as considered object
    bbox = cv2.selectROI(frame)
    print('[INFO] select ROI and press ENTER or SPACE')
    print('[INFO] cancel selection by pressing C')
    #! init tracker with chosen object
    ok = tracker.init(frame, bbox)
    if not ok:
        print('[ERROR] tracker not initialized')
        sys.exit()
    os.system('cls')
    print('[INFO] tracker was initialized on ROI')
    
    #! here we iterate over all frames of video:
    i = -1
    while True:
        ret, frame = video.read()
        if not ret:
            print('[INFO] end of video file reached')
            break
        #! update position of ROI based on tracker prediction
        ok, bbox = tracker.update(frame)
        if ok:
            i += 1
            (x, y, w, h) = [int(v) for v in bbox]
            #! use predicted bounding box coordinates to draw a rectangle using cv2.rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), colours, 10)
            #! here we save frame and bbox coordinates in YOLO format
            with open(f'output/{i}.txt', 'w') as f:
                to_write = f'0 {((x+w)/2)/frame_width} {((y+h)/2)/frame_height} {w/frame_width} {h/frame_height}'
                f.writelines(to_write)
                cv2.imwrite(f'output/{i}.jpg', frame)
            cv2.putText(frame, str(tracker_type), (10, 30), cv2.QT_FONT_NORMAL, 1, (255, 255, 255))

        else:
            #! if prediction failed and no bounding box coordinates are available
            cv2.putText(frame, 'No Track', (10, 30), cv2.QT_FONT_NORMAL, 1, (0, 0, 255))

        cv2.imshow('Single Track', frame)
        video_output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()