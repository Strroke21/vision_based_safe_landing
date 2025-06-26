import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor
import math
from pymavlink import mavutil
import time

device = torch.device("cuda") 

# Load model and processor
model_path = "/home/deathstroke/Documents/dpt_swin2_tiny_256.pt"
state_dict = torch.load(model_path, map_location=device)
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

hfov = 87*(math.pi/180)
vfov = 58*(math.pi/180)

red_boundary_threshold = 0.05
green_area_threshold = 0.8

rng_alt = 0

cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def connect(connection_string):

    vehicle =  mavutil.mavlink_connection(connection_string)

    return vehicle

def target_coords(lat, lon, north_offset_m, east_offset_m, heading_deg):
    """
    lat, lon: current GPS
    north_offset_m: forward offset (from image)
    east_offset_m: right offset (from image)
    heading_deg: drone heading in degrees (0° = North, clockwise)
    """
    # Convert heading to radians
    heading_rad = math.radians(heading_deg)

    # Rotate x, y based on heading
    x_rot = east_offset_m * math.cos(heading_rad) - north_offset_m * math.sin(heading_rad)
    y_rot = east_offset_m * math.sin(heading_rad) + north_offset_m * math.cos(heading_rad)

    R = 6378137  # Earth radius
    d_lat = y_rot / R
    d_lon = x_rot / (R * math.cos(math.radians(lat)))

    tar_lat = lat + math.degrees(d_lat)
    tar_lon = lon + math.degrees(d_lon)

    return tar_lat, tar_lon

def global_position(vehicle):
    while True:
        msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            lat = msg.lat / 1e7  # Convert from int to degrees
            lon = msg.lon / 1e7  
            hdg = msg.hdg / 100  
            return lat, lon, hdg
        
def get_rangefinder_data(vehicle):
    global rng_alt
    # Wait for a DISTANCE_SENSOR or RANGEFINDER message
    while True:
        msg = vehicle.recv_match(type='DISTANCE_SENSOR', blocking=False)
        if msg is not None:
            dist = msg.current_distance # in meters
            if dist is not None:
                rng_alt = dist/100

        return rng_alt
    
def find_safe_spot(frame,red_boundary_threshold, green_area_threshold, altitude, current_lat,current_lon):
    frame = cv2.resize(frame, (640, 480))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = feature_extractor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (640, 480), interpolation=cv2.INTER_CUBIC)
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Threshold to get elevated regions
    threshold = 128
    _, elevated_mask = cv2.threshold(depth_map, threshold, 255, cv2.THRESH_BINARY)
    elevated_mask = elevated_mask.astype(np.uint8)

    # Red boundary around elevated area
    kernel_size = int(min(elevated_mask.shape[:2]) * red_boundary_threshold)
    kernel_size = max(3, kernel_size | 1)  # ensure it's odd and at least 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(elevated_mask, kernel, iterations=1)
    boundary_mask = cv2.subtract(dilated, elevated_mask)

    # Invert elevated mask to get black region
    black_mask = cv2.bitwise_not(cv2.bitwise_or(elevated_mask, boundary_mask))

    # Remove corners from black_mask (10% margin)
    h, w = black_mask.shape
    margin_y = int(0.1 * h)
    margin_x = int(0.1 * w)
    black_mask[:margin_y, :] = 0
    black_mask[-margin_y:, :] = 0
    black_mask[:, :margin_x] = 0
    black_mask[:, -margin_x:] = 0

    # Distance transform from red boundary to black region
    inverted_red = cv2.bitwise_not(boundary_mask)
    distance_map = cv2.distanceTransform(inverted_red, distanceType=cv2.DIST_L2, maskSize=5)

    # Mask distance_map with black region only
    valid_distance = cv2.bitwise_and(distance_map, distance_map, mask=black_mask)

    coords = np.column_stack(np.where(valid_distance > 0))
    distances = valid_distance[valid_distance > 0]

    if len(distances) > 0:
        sorted_indices = np.argsort(-distances)  # sort descending
        sample_count = int(green_area_threshold * len(sorted_indices))
        selected_coords = coords[sorted_indices[:sample_count]]
    else:
        selected_coords = []

    # Create green mask from selected coordinates
    green_mask = np.zeros((h, w), dtype=np.uint8)
    for y, x in selected_coords:
        green_mask[y, x] = 255

    # Smooth the green region
    green_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask_dilated = cv2.dilate(green_mask, green_kernel, iterations=2)

    # Create a BGR image for the final visualization
    elevated_mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    elevated_mask_bgr[elevated_mask == 255] = [255, 255, 255]  # white (elevated)
    elevated_mask_bgr[boundary_mask == 255] = [0, 0, 255]       # red (boundary)
    elevated_mask_bgr[green_mask_dilated == 255] = [0, 255, 0]  # green (flat & safe)

    # Step 1: Get the largest green contour (already done)
    contours, _ = cv2.findContours(green_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea, default=None)

    central_pixel = None
    if max_contour is not None and cv2.contourArea(max_contour) > 0:
        # Step 2: Create full mask of the largest green region
        largest_green_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(largest_green_mask, [max_contour], -1, 255, -1)

        # Step 3: Remove elevated (white) region from it
        valid_safe_area = cv2.bitwise_and(largest_green_mask, cv2.bitwise_not(elevated_mask))

        # Step 4: Clean up with morphological operations (remove noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        valid_safe_area = cv2.erode(valid_safe_area, kernel, iterations=1)

        # Step 5: Distance transform to get most central pixel
        dist_map = cv2.distanceTransform(valid_safe_area, cv2.DIST_L2, 5)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist_map)
        central_pixel = maxLoc
        cx, cy = central_pixel

        # Step 2: Convert pixel to meters
        img_center_x = w / 2
        img_center_y = h / 2

        angle_per_pixel_x = hfov / w
        angle_per_pixel_y = vfov / h

        delta_x = (cx - img_center_x) * angle_per_pixel_x
        delta_y = (cy - img_center_y) * angle_per_pixel_y

        north_meters = math.tan(delta_x) * altitude
        east_meters = -math.tan(delta_y) * altitude

        heading = global_position(vehicle)[2]  

        # Step 3: Draw a blue circle at the central pixel
        cv2.circle(elevated_mask_bgr, central_pixel, 5, (255, 0, 0), -1)

        print(f"Central pixel (px): ({cx}, {cy}) → Coordinates in meters from center: ({east_meters:.2f} m, {north_meters:.2f} m)")

        coords = target_coords(current_lat,current_lon,east_meters,north_meters,heading)
        print(f"Target coordinates (lat, lon): {coords}")

        timestamp = str(int(time.time()))
        cv2.imwrite("depth_" + timestamp + ".png", elevated_mask_bgr)
        cv2.imwrite("original_" + timestamp +".png", frame)
        print("Images saved with timestamp:", timestamp)

        return coords

    else:
        return None

vehicle = connect('/dev/ttyACM0')
counter = 0

while True:
    altitude = get_rangefinder_data(vehicle)
    pos = global_position(vehicle)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    if ret:
        counter += 1
    
    if counter <= 10:
        target_coords = find_safe_spot(frame, red_boundary_threshold, green_area_threshold, altitude, pos[0], pos[1])
    
    elif counter > 10:
        break

