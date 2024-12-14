import cv2 as cv
import time
from io import BytesIO
from BrandDetectionModelClass import BrandLogoDetectionModel
# Assuming the BrandLogoDetectionModel class code is already imported
# Initialize the BrandLogoDetectionModel
model = BrandLogoDetectionModel()

path=r"C:\Users\deban\Programming\Programs\FlipkartGrid\non-updated_file_structs\product_test.mp4"
model.process_from_video(0)
df=model.get_detections_csv()
df.to_csv('Result.csv')
