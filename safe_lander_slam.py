import cv2
import numpy as np
import time
import math
from pymavlink import mavutil

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

MAX_DISTANCE = 10

hfov = 87 * (math.pi/180)
vfov = 58 * (math.pi/180)

width = 848
height = 480

final_alt = 2
flatness = 0.2

landing_velocity = -0.3
disparity_to_depth_scale = 0.0010000000474974513

conn_string = '/dev/ttyACM0'
rng_alt = [0]
counter = 0
safe_spot_dist_min = 1
dist_to_spot = [0]

#square pattern for landing
left_x = -5
backward_y = -5
right_x = 5
forward_y = 5

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

class SafeLander(Node):
    def __init__(self):
        super().__init__('safe_lander')

        self.bridge = CvBridge()
        # Subscriptions
        self.subscription = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
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

            if self.counter == 1:
                while True:
                    dist_to_spot = distance_to_safespot(vehicle, x_dist, y_dist)
                    self.get_logger().info(f"Distance to landing spot: {dist_to_spot:.2f}m")
                    if dist_to_spot <= safe_spot_dist_min:
                        time.sleep(0.1)
                        break

            else:
                pass
        
        if min_diff > flatness and altitude >= final_alt:
            self.search_counter += 1
            if self.search_counter == 1:
                send_position_setpoint(vehicle,left_x,0,0) #move left
            elif self.search_counter == 2:
                send_position_setpoint(vehicle,0,backward_y,0) #move backward
            elif self.search_counter == 3:
                send_position_setpoint(vehicle,right_x,0,0) #move right
            elif self.search_counter == 4:
                send_position_setpoint(vehicle,0,forward_y,0) #move forward
            else:
                self.get_logger().info("No safe landing spot found")
                self.get_logger().info("Performing emergency landing")

        if self.counter > 1:
            send_velocity_setpoint(vehicle, 0, 0, landing_velocity)
            self.get_logger().info(f"Descending to landing spot at {landing_velocity:.2f} m/s")
            time.sleep(0.01)

        if altitude <= final_alt:
            VehicleMode(vehicle, "LAND")
            self.get_logger().info("Landing Mode Activated")
            time.sleep(1)
            self.get_logger().info("Landing Final Altitude Reached")
            self.destroy_node()

        cv2.imshow("Depth Grid", frame_colored)
        cv2.waitKey(1)
    

def main(args=None):
    while True:
        mode = str(flightMode(vehicle))
        if mode=='LAND':
            break
        else:
            print(f"Vehicle is not in LAND mode: {mode}")
            print("Waiting for LAND mode...")

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
