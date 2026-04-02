#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
import math
from pymavlink import mavutil

MAX_DISTANCE = 10.0

hfov, vfov = 87.0, 58.0
flatness, final_alt = 0.2, 2
disparity_to_depth_scale = 0.0010000000474974513
MAX_DISTANCE = 10.0

fcu_addr = '/dev/ttyACM0' #'tcp:127.0.0.1:5763' 
current_target = None

HYST_THRESHOLD = 3   # frames required to switch
DIFF_MARGIN = 0.02   # meters improvement required
search_counter = 0
safe_spot_radius = 2

###### functions #######

def connect(connection_string):

    vehicle =  mavutil.mavlink_connection(connection_string)

    return vehicle

def set_parameter(vehicle, param_name, param_value, param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
    # Send PARAM_SET message to change the parameter
    vehicle.mav.param_set_send(vehicle.target_system,vehicle.target_component,param_name.encode('utf-8'),param_value,param_type)
    #usage set_parameter(vehicle, "PARAM_NAME", 1)

def send_land_message(x_ang,y_ang):
    msg = vehicle.mav.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x_ang,
        y_ang,
        0, #distance to marker (m)
        0, #size_x (rad)
        0, #size_y (rad)
        ) 
    vehicle.mav.send(msg)

def get_system_status(vehicle):
    # Wait for a 'HEARTBEAT' message
    message = vehicle.recv_match(type='HEARTBEAT', blocking=True, timeout=10)

    if message is None:
        raise TimeoutError("Did not receive HEARTBEAT message in time")

    # Interpret the system status
    system_status = message.system_status

    status_mapping = {
        mavutil.mavlink.MAV_STATE_UNINIT: 'Uninitialized',
        mavutil.mavlink.MAV_STATE_BOOT: 'Booting',
        mavutil.mavlink.MAV_STATE_CALIBRATING: 'Calibrating',
        mavutil.mavlink.MAV_STATE_STANDBY: 'Standby',
        mavutil.mavlink.MAV_STATE_ACTIVE: 'Active',
        mavutil.mavlink.MAV_STATE_CRITICAL: 'Critical',
        mavutil.mavlink.MAV_STATE_EMERGENCY: 'Emergency',
        mavutil.mavlink.MAV_STATE_POWEROFF: 'Powered Off',
        mavutil.mavlink.MAV_STATE_FLIGHT_TERMINATION: 'Flight Termination',
    }

    return status_mapping.get(system_status, 'Unknown')

def get_rangefinder_data(vehicle):
    global rng_alt
    while True:
        msg = vehicle.recv_match(type='DISTANCE_SENSOR', blocking=True)
        if msg is not None:
            rng_alt = msg.current_distance/100  # in meters
        return rng_alt

def compute_angle_scale_5x5(altitude, hfov, vfov,
                           base_gain=0.8,
                           min_scale=0.15,
                           max_scale=0.9,
                           smooth_factor=0.75,
                           prev_scale=[0.5]):
    """
    Scale factor for 5x5 grid (using inner 3x3 cells).

    Adjustments:
    - Smaller cells → less aggressive inherently
    - Inner region → reduced effective FOV → compensate
    """

    if altitude <= 0:
        return min_scale

    # --- Convert FOV to radians (IMPORTANT if not already) ---
    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)

    # --- Camera footprint ---
    width_m  = 2 * altitude * math.tan(hfov_rad / 2)
    height_m = 2 * altitude * math.tan(vfov_rad / 2)

    # --- 5x5 grid cell size ---
    cell_w = width_m / 5.0
    cell_h = height_m / 5.0

    avg_cell = 0.5 * (cell_w + cell_h)

    # --- Compensation factor ---
    # Inner usable grid = 3/5 of full → boost response
    compensation = 5.0 / 3.0   # ≈1.67

    # --- Base scaling ---
    scale = base_gain * avg_cell * compensation

    # Normalize (prevents blow-up)
    scale = scale / (1.0 + scale)

    # Clamp
    scale = max(min_scale, min(scale, max_scale))

    # Smooth (EMA)
    scale = smooth_factor * prev_scale[0] + (1 - smooth_factor) * scale
    prev_scale[0] = scale

    return scale
    
def enable_data_stream(vehicle,stream_rate):

    vehicle.wait_heartbeat()
    vehicle.mav.request_data_stream_send(
    vehicle.target_system, 
    vehicle.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    stream_rate,1)

def VehicleMode(vehicle, mode):
    modes = ["STABILIZE","ACRO","ALT_HOLD","AUTO","GUIDED","LOITER","RTL","CIRCLE","","LAND"]
    mode_id = modes.index(mode) if mode in modes else 12
    vehicle.mav.set_mode_send(vehicle.target_system,
                              mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                              mode_id)

def status_check(vehicle):
    while True:
        status = get_system_status(vehicle)
        print("Vehicle State: ",status)
        if ('Critical' in status) or ('Emergency' in status):
            return status

vehicle = connect(fcu_addr)
enable_data_stream(vehicle,stream_rate=200)
    ##SETUP PARAMETERS TO ENABLE PRECISION LANDING
set_parameter(vehicle,'PLND_ENABLED', 1)
set_parameter(vehicle,'PLND_TYPE',1) ##1 for companion computer
set_parameter(vehicle,'PLND_EST_TYPE', 0) # 0 for raw sensor, 1 for kalman filter pos estimation
set_parameter(vehicle,'LAND_SPEED',30) ##Descent speed of 30cm/s
set_parameter(vehicle,'PLND_XY_DIST_MAX', 8)
set_parameter(vehicle,'PLND_LAG',0.0519) #lag with ros2 and rsd455
# Start streaming

class SafeLander(Node):
    def __init__(self):
        super().__init__('safe_lander')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            1
        )

        self.last_valid_altitude = 2.0

        # 🔹 Persistent selected cell (hysteresis)
        self.current_cell = None  # (i, j)
        self.current_diff = float('inf')

    def depth_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

        mask = (frame <= (MAX_DISTANCE * 1000)).astype(np.uint16)
        frame = frame * mask

        height, width = frame.shape

        altitude = get_rangefinder_data(vehicle)

        self.get_logger().info(f"[Altitude: {altitude:.2f} m]")

        # ---- GRID (5x5, ignore edges) ----
        grid_rows, grid_cols = 5, 5
        cell_w, cell_h = width // grid_cols, height // grid_rows

        differences = []
        valid_indices = []

        for i in range(1, grid_rows - 1):
            for j in range(1, grid_cols - 1):
                seg = frame[
                    i * cell_h:(i + 1) * cell_h,
                    j * cell_w:(j + 1) * cell_w
                ]

                seg = seg[seg > 0]

                if seg.size == 0:
                    diff = float('inf')
                else:
                    diff = np.max(seg) - np.min(seg)

                differences.append(diff * disparity_to_depth_scale)
                valid_indices.append((i, j))

        # ---- FIND BEST CELL ----
        min_diff = min(differences)
        min_idx = differences.index(min_diff)
        best_cell = valid_indices[min_idx]

        # ---- HYSTERESIS LOGIC ----
        if self.current_cell is None:
            # First time selection
            self.current_cell = best_cell
            self.current_diff = min_diff

        else:
            # Find diff of current cell
            if self.current_cell in valid_indices:
                idx = valid_indices.index(self.current_cell)
                self.current_diff = differences[idx]
            else:
                self.current_diff = float('inf')

            # 🔴 Switch ONLY if current becomes bad
            if self.current_diff > flatness:
                self.current_cell = best_cell
                self.current_diff = min_diff

        grid_i, grid_j = self.current_cell

        # ---- VISUALIZATION ----
        depth_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Draw grid
        for i in range(1, grid_cols):
            cv2.line(frame_colored, (i * cell_w, 0), (i * cell_w, height), (255, 255, 255), 1)
        for i in range(1, grid_rows):
            cv2.line(frame_colored, (0, i * cell_h), (width, i * cell_h), (255, 255, 255), 1)

        # Annotate
        for idx, diff in enumerate(differences):
            i, j = valid_indices[idx]
            x, y = j * cell_w, i * cell_h
            cv2.putText(frame_colored, f"{diff:.2f}",
                        (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

        # ---- LANDING ----
        if self.current_diff <= flatness and altitude >= final_alt:
            scale_factor = compute_angle_scale_5x5(altitude, math.degrees(hfov), math.degrees(vfov))
            top_left_x = grid_j * cell_w
            top_left_y = grid_i * cell_h
            bottom_right_x = top_left_x + cell_w
            bottom_right_y = top_left_y + cell_h

            x_avg = (top_left_x + bottom_right_x) / 2
            y_avg = (top_left_y + bottom_right_y) / 2

            x_ang = (x_avg - width / 2) * (hfov / width)
            y_ang = (y_avg - height / 2) * (vfov / height)

            x_dist = altitude * np.tan(np.radians(-y_ang)) 
            y_dist = altitude * np.tan(np.radians(x_ang)) 
            send_land_message(x_ang*scale_factor, y_ang*scale_factor)
            self.get_logger().info(
                f"[Landing target → x: {x_dist:.2f}, y: {y_dist:.2f} scale: {scale_factor:.2f}]"
            )

            if altitude <= final_alt:
                self.get_logger().info("Landing Final Altitude Reached")
                self.destroy_node()

            # Draw selected cell
            cv2.rectangle(
                frame_colored,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                (0, 255, 0),
                2
            )

        elif self.current_diff > flatness and altitude >= final_alt:
            self.get_logger().info(f"[No suitable landing zone found. Best diff: {self.current_diff:.2f} m]")
            search_counter += 1
            offset_ang = math.atan(safe_spot_radius/altitude)
            print(f"offset angle: {math.degrees(offset_ang):.2f} degrees")
            if search_counter == 10:
                self.get_logger().info("[Search mode activated: moving right]")
                send_land_message(0, offset_ang)  # Move right
            
            if search_counter == 20:
                self.get_logger().info("[Search mode: moving back]")
                send_land_message(-offset_ang, 0)  # Move back
            
            if search_counter == 30:
                self.get_logger().info("[Search mode: moving left]")
                send_land_message(0, -offset_ang)  # Move left
            
            if search_counter == 40:
                self.get_logger().info("[Search mode: moving forward]")
                send_land_message(offset_ang, 0)  # Move forward
                
            elif search_counter > 40:
                self.get_logger().info("[Performing normal landing descent]")
                send_land_message(0, 0)  
                

        # cv2.imshow("Depth Grid", frame_colored)
        # cv2.waitKey(1)


    def main(args=None):
        status = status_check(vehicle) #input("status:")
        print("Vehicle State: ",status)
        if ('Critical' in status) or ('Emergency' in status):
            
            rclpy.init(args=args)
            node = SafeLander()
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()

    if __name__ == '__main__':
        main()