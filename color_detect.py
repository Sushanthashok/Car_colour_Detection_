# color_detect.py
import cv2
import numpy as np

def is_blue_car_hsv(bgr_roi, blue_thresh=0.08):
    """
    Detects if the given BGR ROI contains a blue car using HSV color masking.
    
    Args:
        bgr_roi: Car crop (OpenCV BGR image)
        blue_thresh: Fraction of blue pixels required to classify as blue
        
    Returns:
        (is_blue: bool, blue_fraction: float)
    """

    if bgr_roi is None or bgr_roi.size == 0:
        return False, 0.0

    h, w = bgr_roi.shape[:2]
    if h < 8 or w < 8:
        return False, 0.0

    # Resize small for stable detection
    roi_small = cv2.resize(bgr_roi, (160, 160), interpolation=cv2.INTER_AREA)

    # Convert to HSV
    hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)

    # Blue HSV range
    lower_blue = np.array([90, 60, 60], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    # Create blue mask
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Calculate % of blue pixels
    blue_fraction = (mask_blue > 0).mean()

    # Decide if car is blue
    is_blue = blue_fraction >= blue_thresh

    return bool(is_blue), float(blue_fraction)
