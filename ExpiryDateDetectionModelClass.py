import re
from datetime import datetime
from io import BytesIO
import time
import cv2 as cv
import numpy as np
import dateutil.parser
import pandas as pd
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity


def validate_date(parsed_date):
    current_year = datetime.now().year
    return (
        current_year - 10 <= parsed_date.year <= current_year + 10
        and 1 <= parsed_date.month <= 12
    )


def parse_date_multiple_methods(date_str):
    try:
        parsed_date = dateutil.parser.parse(date_str,dayfirst=True)
        if validate_date(parsed_date):
            return parsed_date
    except:
        pass
    return None


def find_expiry_date_expression_matching(text):
    date_patterns = [
        r'EXP(?:IRY)?[\s:]*(\d{2}[-/\s]\d{2}[-/\s]\d{2,4})',
        r'EXP(?:IRY)?[\s:]*(\d{2}[-/\s][A-Za-z]{3}[-/\s]\d{2,4})',
        r'BEST[\s]*(?:BEFORE|BY)[\s:]*(\d{2}[-/\s]\d{2}[-/\s]\d{2,4})',
        r'USE[\s]*(?:BEFORE|BY)[\s:]*(\d{2}[-/\s]\d{2}[-/\s]\d{2,4})'
    ]
    for pattern in date_patterns:
        match = re.search(pattern,text,re.IGNORECASE)
        if match:
            parsed_date=parse_date_multiple_methods(match.group(1))
            if parsed_date:
                return parsed_date
    return None

def find_expiry_date_brute(text):
    date_patterns = [r'(\d{2}[-/]\d{2}[-/]\d{2,4})']
    valid_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern,text,re.IGNORECASE)
        for match in matches:
            parsed_date = parse_date_multiple_methods(match)
            if parsed_date:
                valid_dates.append(parsed_date)
    if valid_dates:
        return max(valid_dates)
    else:
        date_pattern = r'(\d{2}[-/]\d{2,4})'
        valid_dates = []
    
        matches = re.findall(date_pattern, text, re.IGNORECASE)
        
        for match in matches:
            normalized_date = match.replace('-', '/')
            date_str = f"01/{normalized_date}"
            try:
                parsed_date = dateutil.parser.parse(date_str,dayfirst=True)
                valid_dates.append(parsed_date)
            except ValueError:
                continue
    return max(valid_dates) if valid_dates else None
    
class ExpiryDateDetectionModel():
    def __init__(self):
        self._name = 'expiry-date-detection'
        self.model = PaddleOCR(lang="en", use_angle_cls=False)
        self.detections = []
        self.latest_detection_time=time.time()
        self.prev_detected='None'
    def name(self):
        return self._name

    def process_frame(self, frame, prev_frame=None, write=True):
     l=frame.shape[0]
     b=frame.shape[1]
     frame=cv.rectangle(frame, (int(b/4),int(l/4)),(int((3*b)/4),int((3*l)/4)),(0,0,255), 2)
     points=np.array([[int(b/4),int(l/4)],[int((3*b)/4),int((l)/4)],[int(3*b/4),int(3*l/4)],[int(b/4),int(3*l/4)]])
     mask=np.zeros_like(frame)
     mask=cv.fillPoly(mask,[points],(255,255,255))
     mask2=cv.bitwise_not(mask)
     n_roi=cv.bitwise_and(frame,mask2)
     roi=cv.bitwise_and(frame,mask)
     frame_curr_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     frame_prev_g = (cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY))
     ssim = structural_similarity(frame_curr_g, frame_prev_g)
     exp_date_res = None
     if ssim <= 0.75:
        result = self.model.ocr(roi, cls=True)
        full_text = ""
        if result and result[0]:
            for res in result[0]:
                text = res[1][0]
                full_text = full_text + " " + text + " "
                npar = np.array(res[0])
                npar = npar.astype(int)
                pos = npar[0]
                text_size, _ = cv.getTextSize(text,cv.FONT_HERSHEY_SIMPLEX,0.6,2)
                text_w, text_h = text_size
                roi = cv.rectangle(roi,pos,[pos[0]+text_w, pos[1]+text_h],(255,255,255), -1)
                cv.putText(roi, text, (pos[0],pos[1]+text_h+-1-1),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255), 2)
        expiry_date_e_m = find_expiry_date_expression_matching(full_text)
        expiry_date_b = find_expiry_date_brute(full_text)
        if expiry_date_b and expiry_date_e_m:
            if expiry_date_e_m != expiry_date_b:
                exp_date_res = max(expiry_date_b, expiry_date_e_m)
        elif expiry_date_e_m:
            exp_date_res = expiry_date_e_m
        elif expiry_date_b:
            exp_date_res = expiry_date_b
        detection_time=time.time()
        if exp_date_res:
          if(detection_time-self.latest_detection_time>20 or self.prev_detected!=exp_date_res):
            if exp_date_res < datetime.now():
                self.detections.append(
                    {
                        "Expiry date": exp_date_res,
                        "Status": "Expired",
                        "Days Remaining": None,
                    }
                )
            else:
                self.detections.append(
                    {
                        "Expiry date": exp_date_res,
                        "Status": "Not Expired",
                        "Days Remaining": (exp_date_res - datetime.now()).days,
                    }
                )
            self.prev_detected=exp_date_res
            self.latest_detection_time=time.time()
     res = cv.bitwise_or(roi, n_roi)
     cv.imshow('Frame',res)
     if(exp_date_res is not None):
         ret=cv.waitKey(3000)
    def process_from_video(self,path):
        # Process video bytes as frames and detect expiry dates in them
        video_stream = cv.VideoCapture(path)
        if not video_stream.isOpened():
            raise ValueError("Failed to open video stream from bytes")
        prev_frame=None
        while True:
            ret, frame = video_stream.read()
            if not ret:
                break  # Exit loop if no more frames
            if prev_frame is not None:
                self.process_frame(frame, prev_frame, write=True)
            prev_frame = frame
            if(cv.waitKey(5) & 0xFF==ord('q')):
                break
        video_stream.release()

    def get_detections_csv(self):
        df = pd.DataFrame(self.detections)
        return df
