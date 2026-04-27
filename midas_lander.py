#!/usr/bin/python3

import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor
import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge 
from pymavlink import mavutil
import os

bridge = CvBridge()

device = torch.device("cuda") 
save_dir = "/home/deathstroke/Desktop/vision_safe_landing/"
os.makedirs(save_dir, exist_ok=True)
# Load model and processor
model_path = "/home/deathstroke/Documents/dpt_swin2_tiny_256.pt"
state_dict = torch.load(model_path, map_location=device)
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
frame_np = None

feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

hfov = 90*(math.pi/180)
vfov = 65*(math.pi/180)

red_boundary_threshold = 0.05
green_area_threshold = 1.0

lander_alt = 15.0

# altitude = 50
# heading = 360

# current_lat = 19.1346473
# current_lon = 72.9108256

def image_callback(msg):
    global frame_np, topic_delay
    frame_np = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    topic_delay = time.time() - msg.header.stamp.sec - msg.header.stamp.nanosec * 1e-9

rclpy.init()
ros_node = rclpy.create_node('Midas_Lander')

image_sub = ros_node.create_subscription(
    Image,
    '/camera/camera/color/image_raw',
    image_callback,
    1)

def target_coords(lat, lon, north_offset_m, east_offset_m, heading_deg):
    heading_rad = math.radians(heading_deg)

    x_rot = east_offset_m * math.cos(heading_rad) - north_offset_m * math.sin(heading_rad)
    y_rot = east_offset_m * math.sin(heading_rad) + north_offset_m * math.cos(heading_rad)

    R = 6378137
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
        
def connect(connection_string):
    while True:
        try:
            vehicle =  mavutil.mavlink_connection(connection_string)

            return vehicle
        except Exception as e:
            print(f"Connection failed: {e}")
            time.sleep(1)  

def enable_data_stream(vehicle,stream_rate):

    vehicle.wait_heartbeat()
    vehicle.mav.request_data_stream_send(
    vehicle.target_system, 
    vehicle.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    stream_rate,1)

def get_rangefinder_data(vehicle):
    global rng_alt
    # Wait for a DISTANCE_SENSOR or RANGEFINDER message
    msg = vehicle.recv_match(type='DISTANCE_SENSOR', blocking=False)
    if msg is not None:
        dist = msg.current_distance # in meters
        if dist is not None:
            rng_alt = dist/100

    return rng_alt

def current_alt(vehicle):
    while True:
        msg = vehicle.recv_match(type='TERRAIN_REPORT', blocking=True)
        if msg:
            curr_alt = msg.current_height #in meters
            return curr_alt

def VehicleMode(vehicle,mode):

    modes = ["STABILIZE", "ACRO", "ALT_HOLD", "AUTO", "GUIDED", "LOITER", "RTL", "CIRCLE","", "LAND"]
    if mode in modes:
        mode_id = modes.index(mode)
    else:
        mode_id = 12
    ##### changing to guided mode #####
    #mode_id = 0:STABILIZE, 1:ACRO, 2: ALT_HOLD, 3:AUTO, 4:GUIDED, 5:LOITER, 6:RTL, 7:CIRCLE, 9:LAND 12:None
    vehicle.mav.set_mode_send(
        vehicle.target_system,mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,mode_id)
    
def send_position_setpoint(vehicle, pos_x, pos_y, pos_z,FRAME):

    # Send MAVLink command to set position
    vehicle.mav.set_position_target_local_ned_send(
        0,                          # time_boot_ms (not used)
        vehicle.target_system,       # target_system
        vehicle.target_component,    # target_component
        FRAME,  # frame
        0b110111111000,        # type_mask (only for postion)
        pos_x, pos_y, pos_z,   # position 
        0, 0, 0,                 # velocity in m/s (not used)
        0, 0, 0,                    # acceleration (not used)
        0, 0                        # yaw, yaw_rate (not used)
    )

def set_parameter(vehicle, param_name, param_value, param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    # Send PARAM_SET message to change the parameter
    vehicle.mav.param_set_send(vehicle.target_system,vehicle.target_component,param_name.encode('utf-8'),param_value,param_type)
    #usage set_parameter(vehicle, "PARAM_NAME", 1)

def find_safe_spot(frame,red_boundary_threshold, green_area_threshold, altitude, current_lat,current_lon, heading):
    frame = cv2.resize(frame, (640, 480))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = feature_extractor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (640, 480), interpolation=cv2.INTER_CUBIC)
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ================= CANNY EDGE GUIDANCE =================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)

    edges_mask = edges_dilated / 255.0

    depth_map = depth_map * (0.5 + 0.5 * edges_mask)
    depth_map = depth_map.astype(np.uint8)
    # ======================================================

    # Threshold to get elevated regions
    threshold = 128
    _, elevated_mask = cv2.threshold(depth_map, threshold, 255, cv2.THRESH_BINARY)
    elevated_mask = elevated_mask.astype(np.uint8)

    # Red boundary
    kernel_size = int(min(elevated_mask.shape[:2]) * red_boundary_threshold)
    kernel_size = max(3, kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(elevated_mask, kernel, iterations=1)
    boundary_mask = cv2.subtract(dilated, elevated_mask)

    # Black region
    black_mask = cv2.bitwise_not(cv2.bitwise_or(elevated_mask, boundary_mask))

    h, w = black_mask.shape
    margin_y = int(0.025 * h)
    margin_x = int(0.025* w)
    black_mask[:margin_y, :] = 0
    black_mask[-margin_y:, :] = 0
    black_mask[:, :margin_x] = 0
    black_mask[:, -margin_x:] = 0

    # Distance transform
    inverted_red = cv2.bitwise_not(boundary_mask)
    distance_map = cv2.distanceTransform(inverted_red, distanceType=cv2.DIST_L2, maskSize=5)

    valid_distance = cv2.bitwise_and(distance_map, distance_map, mask=black_mask)

    coords = np.column_stack(np.where(valid_distance > 0))
    distances = valid_distance[valid_distance > 0]

    if len(distances) > 0:
        sorted_indices = np.argsort(-distances)
        sample_count = int(green_area_threshold * len(sorted_indices))
        selected_coords = coords[sorted_indices[:sample_count]]
    else:
        selected_coords = []

    green_mask = np.zeros((h, w), dtype=np.uint8)
    for y, x in selected_coords:
        green_mask[y, x] = 255

    green_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask_dilated = cv2.dilate(green_mask, green_kernel, iterations=2)

    elevated_mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    elevated_mask_bgr[elevated_mask == 255] = [255, 255, 255]
    elevated_mask_bgr[boundary_mask == 255] = [0, 0, 255]
    elevated_mask_bgr[green_mask_dilated == 255] = [0, 255, 0]

    contours, _ = cv2.findContours(green_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea, default=None)

    if max_contour is not None and cv2.contourArea(max_contour) > 0:
        largest_green_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(largest_green_mask, [max_contour], -1, 255, -1)

        valid_safe_area = cv2.bitwise_and(largest_green_mask, cv2.bitwise_not(elevated_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        valid_safe_area = cv2.erode(valid_safe_area, kernel, iterations=1)

        dist_map = cv2.distanceTransform(valid_safe_area, cv2.DIST_L2, 5)
        _, _, _, maxLoc = cv2.minMaxLoc(dist_map)
        cx, cy = maxLoc

        img_center_x = w / 2
        img_center_y = h / 2

        angle_per_pixel_x = hfov / w
        angle_per_pixel_y = vfov / h

        delta_x = (cx - img_center_x) * angle_per_pixel_x
        delta_y = (cy - img_center_y) * angle_per_pixel_y

        north_meters = math.tan(delta_x) * altitude
        east_meters = -math.tan(delta_y) * altitude

        cv2.circle(elevated_mask_bgr, (cx, cy), 5, (255, 0, 0), -1)

        print(f"Central pixel: ({cx}, {cy}) → Target posXY ({east_meters:.2f}m, {north_meters:.2f}m)")

        coords = target_coords(current_lat,current_lon,east_meters,north_meters,heading)
        print(f"Target GPS: {coords}")
        cv2.imwrite(os.path.join(save_dir, f"elevated_mask_{int(time.time())}.png"), elevated_mask_bgr)
        cv2.imwrite(os.path.join(save_dir, f"raw_image_{int(time.time())}.png"), frame)
        time.sleep(0.1)
        print("Image saved")
        return east_meters, north_meters, coords

    # cv2.imshow('Original Frame', frame)
    # cv2.imshow('Depth Segmentation', elevated_mask_bgr)
    # cv2.waitKey(0)

def distance_between(current_lat, current_lon, target_lat, target_lon):
            
    R = 6378137  # Earth radius in meters
    dlat = np.radians(current_lat - target_lat)
    dlon = np.radians(current_lon - target_lon)
    a = math.sin(dlat / 2)**2 + math.cos(np.radians(target_lat)) * math.cos(np.radians(current_lat)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance #in meters

vehicle = connect('tcp:127.0.0.1:5763')
enable_data_stream(vehicle, 100)
counter = 0
while True:
    rclpy.spin_once(ros_node, timeout_sec=0.01)
    altitude = current_alt(vehicle) #get_rangefinder_data(vehicle)
    current_lat, current_lon, heading = global_position(vehicle)
    mode = vehicle.flightmode
    print(f"Altitude: {altitude:.2f}m, Lat: {current_lat:.7f}, Lon: {current_lon:.7f}, Heading: {heading:.2f}° mode: {mode}")
    if altitude<=lander_alt and mode=="LAND":
        if frame_np is not None:
            counter+=1
            if counter==1:
                x,y,coords = find_safe_spot(frame_np, red_boundary_threshold, green_area_threshold, altitude, current_lat, current_lon,heading)
                VehicleMode(vehicle,"GUIDED")
                set_parameter(vehicle,"WP_YAW_BEHAVIOR",0)
                time.sleep(0.1)
                send_position_setpoint(vehicle, x, y, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED)
                while True:
                    current_lat, current_lon, heading = global_position(vehicle)
                    dist_to_target = distance_between(current_lat, current_lon, coords[0], coords[1])
                    print(f"Distance to target: {dist_to_target:.2f}m")
                    if dist_to_target < 1.0:
                        VehicleMode(vehicle,"LAND")
                        print("Vehicle in LAND mode)")
                        set_parameter(vehicle,"WP_YAW_BEHAVIOR",1)
                        arming_status = vehicle.motors_armed()
                        if arming_status!= 128:
                            print("Motors disarmed")
                            break
