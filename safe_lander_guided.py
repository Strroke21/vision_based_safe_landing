import cv2
import numpy as np
from openvino.runtime import Core
import time
import math
from pymavlink import mavutil
import pyrealsense2 as rs
# Load OpenVINO Inference Engine
ie = Core()

# Load the network
model_xml = "/home/deathstroke/Desktop/vision_safe_landing/openvino_midas_v21_small_256.xml"
model_bin = "/home/deathstroke/Desktop/vision_safe_landing/openvino_midas_v21_small_256.bin"
network = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(network, "CPU")

# Get input and output tensor names
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

cap = cv2.VideoCapture(4)
cap.set(3, 640)
cap.set(4, 480)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

MAX_DISTANCE = 8.0

hfov = 87 * (math.pi/180)
vfov = 58 * (math.pi/180)

final_alt = 2
flatness = 0.2

landing_velocity = -0.3

conn_string = 'tcp:127.0.0.1:5763' #'/dev/ttyACM0'
rng_alt = [0]
counter = 0
safe_spot_dist_min = 1
dist_to_spot = [0]
###### functions #######

def connect(connection_string):

    vehicle =  mavutil.mavlink_connection(connection_string)

    return vehicle

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
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
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
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
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
    # Wait for a DISTANCE_SENSOR or RANGEFINDER message
    msg = vehicle.recv_match(type='DISTANCE_SENSOR', blocking=False)
    if msg is not None:
        dist = msg.current_distance # in meters
        if dist is not None:
            rng_alt[0] = dist/100

    return rng_alt[0]
            

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

def distance_to_safespot(vehicle, x_dist, y_dist):
    # Get the drone's local position (NED frame)
    pos_msg = vehicle.recv_match(type='LOCAL_POSITION_NED', blocking=True)
    attitude_msg = vehicle.recv_match(type='ATTITUDE', blocking=True)
    
    if pos_msg and attitude_msg:
        pos_x = pos_msg.x  # Drone's X position in local NED
        pos_y = pos_msg.y  # Drone's Y position in local NED
        
        yaw = attitude_msg.yaw  # Yaw in radians (NED frame, CCW positive)

        # Convert from body frame to local NED frame
        x_ned = pos_x + (x_dist * math.cos(yaw) - y_dist * math.sin(yaw))
        y_ned = pos_y + (x_dist * math.sin(yaw) + y_dist * math.cos(yaw))
        
        # Compute the Euclidean distance
        distance = math.hypot(x_ned - pos_x, y_ned - pos_y)
        dist_to_spot[0] = distance

    return dist_to_spot[0]

def safe_spot_finder():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            continue
        
        altitude = get_rangefinder_data(vehicle)
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]
        third_height = height // 3
        third_width = width // 3

        # Process depth map
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))  # Model expects 256x256 input
        img = img.astype(np.float32) / 255.0  # Normalize
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # Change shape to (1,3,256,256)

        # Perform inference
        result = compiled_model([img])[output_layer]
        depth_map = result.squeeze()
        depth_map = cv2.resize(depth_map, (640, 480), interpolation=cv2.INTER_CUBIC)
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Segment the image based on depth
        threshold = 128
        _, segmented_img = cv2.threshold(depth_map, threshold, 255, cv2.THRESH_BINARY)

        # Convert segmented image to color for visualization
        segmented_img_color = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)

        # Draw grid lines (white)
        for i in range(1, 3):
            cv2.line(segmented_img_color, (0, i * third_height), (width, i * third_height), (255, 255, 255), 2)
            cv2.line(segmented_img_color, (i * third_width, 0), (i * third_width, height), (255, 255, 255), 2)

        # Define grid priority (center and direct neighbors first)
        priority_order = [
            (1, 1), (1, 2), (2, 1), (2, 2),  # Center and direct neighbors
            (0, 1), (1, 0), (1, 2), (2, 0), (2, 2), (2, 1),  # Outer neighbors
            (0, 0), (0, 2), (2, 0), (2, 2)  # Corners
        ]

        selected_block = None
        max_black_pixel_count = 0

        # Search for the best block to select
        for i, j in priority_order:
            x_start, y_start = j * third_width, i * third_height
            x_end, y_end = x_start + third_width, y_start + third_height
            block = segmented_img[y_start:y_end, x_start:x_end]
            black_pixel_count = np.sum(block == 0)

            if black_pixel_count > 0:
                if black_pixel_count == block.size:
                    selected_block = (x_start, y_start, x_end, y_end)
                    break  # Fully black block found, stop searching
                elif black_pixel_count > max_black_pixel_count:
                    max_black_pixel_count = black_pixel_count
                    selected_block = (x_start, y_start, x_end, y_end)

        # Highlight the selected block with a red rectangle
        if selected_block:
            counter += 1
            x_start, y_start, x_end, y_end = selected_block
            x_avg = (x_start + x_end) / 2
            y_avg = (y_start + y_end) / 2
            x_ang = (x_avg - width / 2) * (hfov / width)
            y_ang = ((height / 2) - y_avg) * (vfov / height)
            x_dist = altitude * np.tan(x_ang)
            y_dist = altitude * np.tan(y_ang)
            if counter == 1:
                send_position_setpoint(vehicle, x_dist, y_dist, 0)
                print(f" x_dist: {x_dist}, y_dist: {y_dist}")
                while True:
                    dist_to_landing_spot = distance_to_safespot(vehicle, x_dist, y_dist)
                    print(f" Distance to Safe Location: {dist_to_landing_spot:.2f} [m]")
                    if dist_to_landing_spot <= safe_spot_dist_min:
                        break

            cv2.rectangle(segmented_img_color, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)
            break

        # Display results
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Segmented Image with Grid', segmented_img_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def safe_lander():
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        frame = np.asanyarray(depth_frame.get_data())

        # Mask out values greater than MAX_DISTANCE
        mask = frame * depth_frame.get_units() <= MAX_DISTANCE
        frame = frame * mask

        # Get frame dimensions
        height, width = frame.shape

        # Get the altitude from the rangefinder
        #altitude = abs(int(get_local_position(vehicle)[2]))
        altitude = get_rangefinder_data(vehicle)

        # Define the grid size (e.g., 3x3 grid)
        grid_rows = 3
        grid_cols = 3

        # Convert grid cell sizes to pixel dimensions
        third_width = int(width / grid_cols)
        third_height = int(height / grid_rows)

        # Split the frame into dynamic grid cells
        segments = [
            frame[i * third_height:(i + 1) * third_height, j * third_width:(j + 1) * third_width]
            for i in range(grid_rows)
            for j in range(grid_cols)
        ]

        # Process each grid segment
        min_diffs = []
        for segment in segments:
            min_disp = np.min(segment)
            max_disp = np.max(segment)
            min_diffs.append(max_disp - min_disp)

        # Convert disparity differences to meters
        disparity_to_depth_scale = depth_frame.get_units()
        differences_in_meters = [diff * disparity_to_depth_scale for diff in min_diffs]

        # Find the segment with the lowest difference
        min_diff = min(differences_in_meters)
        min_diff_index = differences_in_meters.index(min_diff)

        # Apply color map to the original frame for visualization
        frame_colored = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.03), cv2.COLORMAP_JET)

        # Draw grid lines dynamically
        for i in range(1, grid_cols):
            cv2.line(frame_colored, (i * third_width, 0), (i * third_width, height), (255, 255, 255), 2)
        for i in range(1, grid_rows):
            cv2.line(frame_colored, (0, i * third_height), (width, i * third_height), (255, 255, 255), 2)

        # Annotate each segment with the difference in meters
        for i, diff in enumerate(differences_in_meters):
            top_left_x = (i % grid_cols) * third_width
            top_left_y = (i // grid_cols) * third_height
            cv2.putText(frame_colored, f"{diff:.2f}m", (top_left_x + 5, top_left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Check if the lowest difference is less than or equal to 0.2m
        if min_diff <= flatness and altitude >= final_alt:

            top_left_x = (min_diff_index % 3) * third_width
            top_left_y = (min_diff_index // 3) * third_height
            bottom_right_x = top_left_x + third_width
            bottom_right_y = top_left_y + third_height

            x_avg = (top_left_x + bottom_right_x) / 2
            y_avg = (top_left_y + bottom_right_y) / 2
            x_ang = (x_avg - width / 2) * (hfov / width)
            y_ang = ((height / 2) - y_avg) * (vfov / height)

            x_dist = altitude * np.tan(x_ang)
            y_dist = altitude * np.tan(y_ang)
            
            send_position_setpoint(vehicle, x_dist, y_dist, 0)
            send_velocity_setpoint(vehicle, 0, 0, landing_velocity)
            print(f" x_dist: {x_dist}, y_dist: {y_dist}")

            # Highlight the selected grid cell
            cv2.rectangle(frame_colored, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            if altitude <= final_alt:
                VehicleMode(vehicle,"LAND")
                print("Landing Final Altitude Reached")
                time.sleep(1)
                break

        # Display the frame with the grid and bounding box (if any)
        cv2.imshow("Disparity Grid", frame_colored)
        if cv2.waitKey(1) == ord('q'):
            break


######### Main #########

vehicle = connect(conn_string)
print(f"Vehicle connected: {vehicle}")
enable_data_stream(vehicle, stream_rate=100)

while True:
    mode = flightMode(vehicle)
    if mode=='LAND':
        break
    else:
        print(f"Vehicle is not in LAND mode: {mode}")
        print("Waiting for LAND mode...")

VehicleMode(vehicle, 'GUIDED')
print("Vehicle in GUIDED mode")
time.sleep(1)

safe_spot_finder()

# Start realsense pipeline
pipeline.start(config)

safe_lander()