import cv2
import numpy as np
import time
import math
from pymavlink import mavutil
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from transformers import DPTForDepthEstimation, DPTImageProcessor
from math import radians, cos, sin, sqrt, atan2


### Cuda Parameters ###
device = torch.device("cuda") 

# Load model and processor
model_path = "/home/deathstroke/Documents/dpt_swin2_tiny_256.pt"
state_dict = torch.load(model_path, map_location=device)
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
red_boundary_threshold = 0.05 # Percentage of red boundary as buffer around elevated area
green_area_threshold = 0.8 # Percentage of green area to consider for safe landing

MAX_DISTANCE = 10

hfov = 90 * (math.pi/180)
vfov = 65 * (math.pi/180)

final_alt = 2
flatness = 0.2
lander_alt = 30 #meters

landing_velocity = 0.3 #positive (down) negative (up)
disparity_to_depth_scale = 0.0010000000474974513

conn_string = '/dev/ttyACM0'
rng_alt = 0
counter = 0
safe_spot_dist_min = 2

###### functions #######
def goto_waypoint(vehicle,latitude, longitude, altitude):
    msg = vehicle.mav.set_position_target_global_int_encode(
        time_boot_ms=10,
        target_system=vehicle.target_system,       # Target system (usually 1 for drones)
        target_component=vehicle.target_component,    # Target component (usually 1 for drones)
        coordinate_frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # Frame of reference for the coordinate system
        type_mask=0b0000111111111000,        # Bitmask to indicate which dimensions should be ignored (0b0000111111111000 means all ignored except position)
        lat_int=int(latitude * 1e7),       # Latitude in degrees * 1e7 (to convert to integer)
        lon_int=int(longitude * 1e7),      # Longitude in degrees * 1e7 (to convert to integer)
        alt=altitude,           # Altitude in meters (converted to millimeters)
        vx=0,                         # X velocity in m/s (not used)
        vy=0,                         # Y velocity in m/s (not used)
        vz=0,                         # Z velocity in m/s (not used)
        afx=0, afy=0, afz=0,                   # Accel x, y, z (not used)
        yaw=0, yaw_rate=0                       # Yaw and yaw rate (not used)
    )
    vehicle.mav.send(msg)

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
        print("No safe landing spot found.")
        return global_position(vehicle)  


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

def send_velocity_setpoint(vehicle, vx, vy, vz):

    # Send MAVLink command to set velocity
    vehicle.mav.set_position_target_local_ned_send(
        0,                          # time_boot_ms (not used)
        vehicle.target_system,       # target_system
        vehicle.target_component,    # target_component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b0000111111000111,        # type_mask (only vx, vy, vz, yaw_rate)
        0, 0, 0,                    # position (not used)
        vx, vy, vz,                 # velocity in m/s
        0, 0, 0,                    # acceleration (not used)
        0, 0                        # yaw, yaw_rate (not used)
    )

def send_position_setpoint(vehicle, pos_x, pos_y, pos_z):

    # Send MAVLink command to set velocity
    vehicle.mav.set_position_target_local_ned_send(
        0,                          # time_boot_ms (not used)
        vehicle.target_system,       # target_system
        vehicle.target_component,    # target_component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b110111111000,        # type_mask (only for postion)
        pos_x, pos_y, pos_z,   # position 
        0, 0, 0,                 # velocity in m/s (not used)
        0, 0, 0,                    # acceleration (not used)
        0, 0                        # yaw, yaw_rate (not used)
    )


def send_distance_message(vehicle,z):
    msg = vehicle.mav.distance_sensor_encode(
        0, #time sync system boot !not used
        20, #minimum distance
        7000, #max distance
        z, #current distance must be integer
        0, #type=raw camera !not used
        0, #onboard id !not used
        mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270, #camera facing down
        0
    )
    vehicle.mav.send(msg)

def get_rangefinder_data(vehicle):
    global rng_alt
    # Wait for a DISTANCE_SENSOR or RANGEFINDER message
    msg = vehicle.recv_match(type='DISTANCE_SENSOR', blocking=False)
    if msg is not None:
        dist = msg.current_distance # in meters
        if dist is not None:
            rng_alt = dist/100

    return rng_alt
            

def set_parameter(vehicle, param_name, param_value, param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    # Send PARAM_SET message to change the parameter
    vehicle.mav.param_set_send(vehicle.target_system,vehicle.target_component,param_name.encode('utf-8'),param_value,param_type)

def status_check(vehicle):
    msg = vehicle.recv_match(type='HEARTBEAT', blocking=True)
    if msg:
        info = msg.system_status
        severities = ['UNKNOWN', 'BOOT', 'CALIBRATING', 'STANDBY', 'ACTIVE', 'CRITICAL', 'EMERGENCY', 'POWEROFF', 'TERMINATING']
        return severities[info]

def get_local_position(vehicle):
    msg = vehicle.recv_match(type='LOCAL_POSITION_NED', blocking=True)
    pos_x = msg.x # meters
    pos_y = msg.y  # meters
    pos_z = msg.z  # Meters
    vx = msg.vx
    vy = msg.vy
    vz = msg.vz
    return [pos_x,pos_y,pos_z,vx,vy,vz]

def flightMode(vehicle):
    vehicle.recv_match(type='HEARTBEAT', blocking=True)
    # Wait for a 'HEARTBEAT' message
    mode = vehicle.flightmode
    return mode

def distance_between(current_lat, current_lon, leader_lat,leader_lon):    
    R = 6378137 # Earth radius in meters
    dlat = radians(current_lat - leader_lat)
    dlon = radians(current_lon - leader_lon)
    a = sin(dlat / 2)**2 + cos(radians(leader_lat)) * cos(radians(current_lat)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance 

class SafeLander(Node):
    def __init__(self):
        super().__init__('safe_lander')

        self.bridge = CvBridge()
        # Subscriptions
        self.subscription = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.camera_callback, 10)
        self.counter = 0
        self.search_counter = 0

    def depth_callback(self, msg):
        """Callback for processing depth images and finding safe landing spots."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

        mask = (frame <= (MAX_DISTANCE * 1000)).astype(np.uint16)  # Convert meters to mm
        frame = frame * mask
        
        height, width = frame.shape
        altitude = get_rangefinder_data(vehicle)

        grid_rows, grid_cols = 3, 3
        third_width, third_height = width // grid_cols, height // grid_rows

        # Split frame into 3x3 segments
        segments = [
            frame[i * third_height:(i + 1) * third_height, j * third_width:(j + 1) * third_width]
            for i in range(grid_rows)
            for j in range(grid_cols)
        ]

        min_diffs = []
        for segment in segments:
            min_disp = np.min(segment)
            max_disp = np.max(segment)
            min_diffs.append(max_disp - min_disp)

        # Convert disparity differences to meters
        differences_in_meters = [diff * disparity_to_depth_scale for diff in min_diffs]  # Convert mm to meters

        # Find the flattest segment
        min_diff = min(differences_in_meters)
        min_diff_index = differences_in_meters.index(min_diff)

        # Normalize depth values for visualization
        depth_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Draw grid lines
        for i in range(1, grid_cols):
            cv2.line(frame_colored, (i * third_width, 0), (i * third_width, height), (255, 255, 255), 2)
        for i in range(1, grid_rows):
            cv2.line(frame_colored, (0, i * third_height), (width, i * third_height), (255, 255, 255), 2)

        # Annotate each segment with the difference in meters
        for i, diff in enumerate(differences_in_meters):
            top_left_x = (i % grid_cols) * third_width
            top_left_y = (i // grid_cols) * third_height
            cv2.putText(frame_colored, f"{diff:.2f}m", (top_left_x + 5, top_left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.get_logger().info(f"Differences in meters: {differences_in_meters}")

        if min_diff <= flatness and altitude >= final_alt:
            self.counter += 1
            top_left_x = (min_diff_index % 3) * third_width
            top_left_y = (min_diff_index // 3) * third_height
            bottom_right_x = top_left_x + third_width
            bottom_right_y = top_left_y + third_height

            x_avg = (top_left_x + bottom_right_x) / 2
            y_avg = (top_left_y + bottom_right_y) / 2
            # Compute landing coordinates
            x_ang = (x_avg - width / 2) * (hfov / width)
            y_ang = ((height / 2) - y_avg) * (vfov / height)
            x_dist = altitude * np.tan(np.radians(x_ang))
            y_dist = altitude * np.tan(np.radians(y_ang))

            self.get_logger().info(f"Landing at x: {x_dist:.2f}, y: {y_dist:.2f}")
            send_position_setpoint(vehicle, x_dist, y_dist, 0)
            cv2.rectangle(frame_colored, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)  # Green box


        if (self.counter > 1) and (altitude >= final_alt):
            send_velocity_setpoint(vehicle, 0, 0, landing_velocity)
            self.get_logger().info(f"Descending to landing spot at {landing_velocity:.2f} m/s")
            time.sleep(0.01)

        if altitude <= final_alt:
            VehicleMode(vehicle, "LAND")
            self.get_logger().info("Landing Mode Activated")
            time.sleep(1)
            self.get_logger().info("Landing Final Altitude Reached")
            self.destroy_node()

        #cv2.imshow("Depth Grid", frame_colored)
        #cv2.waitKey(1)

    def camera_callback(self, msg):
        self.search_counter += 1
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        altitude = get_rangefinder_data(vehicle)
        pos = global_position(vehicle)
        if self.search_counter == 1:
            target_coords = find_safe_spot(frame, red_boundary_threshold, green_area_threshold, altitude, pos[0], pos[1])
            goto_waypoint(vehicle, target_coords[0], target_coords[1], altitude)
            dist = distance_between(pos[0], pos[1], target_coords[0], target_coords[1])
            while dist > safe_spot_dist_min:
                pos = global_position(vehicle)
                dist = distance_between(pos[0], pos[1], target_coords[0], target_coords[1])
                self.get_logger().info(f"Distance to target: {dist:.2f} m")
                time.sleep(0.1)  
        else:
            pass

def main(args=None):
    while True:
        mode = str(flightMode(vehicle))
        altitude = get_rangefinder_data(vehicle)
        if (mode=='LAND') and (altitude <= lander_alt):
            break
        else:
            print(f"Current mode: {mode} Curent altitude: {altitude:.2f} m")

    VehicleMode(vehicle, 'GUIDED')
    print("Vehicle in GUIDED mode")
    time.sleep(1)

    rclpy.init(args=args)
    node = SafeLander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    vehicle = connect(conn_string)
    print(f"Vehicle connected: {vehicle}")
    enable_data_stream(vehicle, stream_rate=100)
    main()


