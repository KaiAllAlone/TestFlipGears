import tempfile
from io import BytesIO

import cv2 as cv
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from ultralytics import YOLO
from datetime import datetime 

# from models.base_model import BaseModel, DetectionModels


class CountingModel():
    def __init__(self):
        self._name = "count-model"
        self.model = YOLO(r"UpdatedFileStructs\best.pt")
        self.detections = []
        self.count=0
        self.timestamp=0

    def name(self):
        return self._name

    def process_frame(self, frame, prev_frame=None, write=True):

        # Convert frames to grayscale for SSIM calculation
        curr_frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

        # Compute SSIM to check frame similarity
        ssim = structural_similarity(curr_frame_gray, prev_frame_gray)

        if ssim >= 0.99:
            results = self.model(frame, stream=True, max_det=100)
            self.count = 0
            for r in results:
                for box in r.boxes:
                    [x1, y1, x2, y2] = box.xyxy[0]
                    self.count += 1
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    if write:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.timestamp=datetime.now()
        # return frame
        cv.imshow('Frame',frame)

    def process_from_video(self,path):
        # Write video bytes to a temporary file
        # with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        #     temp_video.write(video_bytes)
        #     temp_video.flush()

            video_stream = cv.VideoCapture(path)

            if not video_stream.isOpened():
                raise ValueError("Failed to open video stream from bytes")
            prev_frame = None
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break
                if prev_frame is not None:
                    self.process_frame(frame, prev_frame, write=True)
                prev_frame = frame
                if(cv.waitKey(5)&0xFF==ord('q')):
                    break
            video_stream.release()

    def get_detections_csv(self) -> BytesIO:
        df = pd.DataFrame({'Timstamp':[self.timestamp],'Quantity':[self.count]})
        # csv_buffer = BytesIO()
        # df.to_csv(csv_buffer, index=False)
        # csv_buffer.seek(0)
        return df
