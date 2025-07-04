#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_apps) repo
# (see https://https://github.com/nepi-engine/nepi_apps)
#
# License: nepi applications are licensed under the "Numurus Software License", 
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment block.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com
#

import os
#### ROS namespace setup
#NEPI_BASE_NAMESPACE = '/nepi/s2x/'
#os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script

import time
import sys
import copy
import threading
import statistics
import numpy as np
import cv2

from nepi_sdk import nepi_sdk 
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_img

from std_msgs.msg import UInt8, Int32, Float32, Bool, Empty, String, Header
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image

from nepi_interfaces.msg import PanTiltLimits, PanTiltPosition, SingleAxisTimedMove, PTXStatus, StringArray
from nepi_interfaces.srv import PTXCapabilitiesQuery

from nepi_interfaces.msg import BoundingBox, BoundingBoxes, ObjectCount, RangeWindow
from nepi_interfaces.msg import AiDetectorInfo, AiDetectorStatus
from nepi_interfaces.srv import AiDetectorInfoQuery, AiDetectorInfoQueryRequest, AiDetectorInfoQueryResponse

from nepi_app_ai_pt_tracker.msg import AiPtTrackerStatus , TrackingErrors


from nepi_api.node_if import NodeSubscribersIF, NodeClassIF
from nepi_api.messages_if import MsgIF
from nepi_api.system_if import SaveDataIF, StatesIF


from nepi_api.connect_mgr_if_ai_model import ConnectMgrAiModelIF
from nepi_api.data_if import ImageIF

from nepi_api.connect_mgr_if_system import ConnectMgrSystemServicesIF


APP_PACKAGE_NAME = 'nepi_app_ai_pt_tracker'
APP_IMG_PUB_FILE = 'ai_pt_tracker_app_img_pub_node.py'

API_LIB_FOLDER = "/opt/nepi/ros/lib/nepi_api"

#########################################
# Node Class
#########################################

class pantiltTargetTrackerApp(object):

  IMAGE_DATA_PRODUCT = 'tracking_image'

  UPDATER_PROCESS_DELAY = 1



  PTX_MAX_TRACK_SPEED_RATIO = 1.0
  PTX_MIN_TRACK_SPEED_RATIO = 0.1
  PTX_OBJ_CENTERED_BUFFER_RATIO = 0.15 # Hysteresis band about center of image for tracking purposes



# Init Settings

  MIN_MAX_RATE = 1
  MAX_MAX_RATE = 10
  FACTORY_MAX_PROC_RATE = 5
  FACTORY_MAX_IMG_RATE = FACTORY_MAX_PROC_RATE * 2

  FACTORY_SCAN_UPDATE_DELAY = 0.5

  FACTORY_FOV_VERT_DEG=70 # Camera Vertical Field of View (FOV)
  FACTORY_FOV_HORZ_DEG=110 # Camera Horizontal Field of View (FOV)

  FACTORY_LABELS_OVERLAY = True
  FACTORY_CLF_OVERLAY = False
  FACTORY_IMG_OVERLAY = False

  FACTORY_MIN_AREA_RATIO = 0.01 # Filters background targets.
  FACTORY_SCAN_SPEED_RATIO = 0.6
  FACTORY_SCAN_TILT_DEG = 0.0

  FACTORY_MIN_MAX_PAN_ANGLES = [-60,60]
  FACTORY_MIN_MAX_TILT_ANGLES = [-45,45]

  FACTORY_TRACK_SPEED_RATIO = 0.6
  FACTORY_TRACK_TILT_OFFSET_DEG = 0.0

  MIN_MAX_ERROR_GOAL = [1,20]
  FACTORY_ERROR_GOAL_DEG = 10

  FACTORY_TARGET_Q_LEN = 3
  FACTORY_TARGET_L_LEN = 5

  node_if = None

  data_products = ["bounding_boxes", IMAGE_DATA_PRODUCT]
  targeting_status_msg = None

  dets_dict = dict()
  dets_names = []
  det_if = None
  current_det = "None"

  current_image_topic = ""
  last_image_topic = ""

  image_sub = None
  img_width = 0
  img_height = 0 

  dets_info_dict = dict()
  detectors_list = []
  classes_list = []
  target_detected = False
  detector_connected = False
  img_source_topic = "None"

  selected_pantilt = "None"
  pt_connected = False
  has_position_feedback = True
  has_adjustable_speed = False
  has_auto_pan = False
  has_auto_tilt = False
  last_sel_pt = ""
  current_scan_dir = 1
  
  pt_status_topic = ""
  pt_status_sub = None
  send_pt_home_pub = None
  set_pt_speed_ratio_pub  = None
  set_pt_position_pub  = None
  set_pt_pan_ratio_pub  = None
  set_pt_tilt_ratio_pub  = None
  set_pt_pan_jog_pub  = None
  set_pt_tilt_jog_pub  = None
  set_pt_soft_limits_pub  = None
  pt_stop_motion_pub  = None
  cur_speed_ratio = 0.5

  last_scan_goal = PanTiltPosition()

  last_track_time = nepi_utils.get_time()

  class_selected = False
  selected_class = "None"
  last_track_dir = -1
  last_pan_pos = 0
  is_moving = False

  no_object_count = 0
  lost_target_count = 0

  reset_image_topic = False
  enabled = False
  app_msg = "Targeting not enabled"
  last_app_msg = ""

  img_msg = None
  img_msg_lock = threading.Lock()

  target_box = None
  target_box_lock = threading.Lock()

  target_box_q = []
  target_box_q_lock = threading.Lock()
  target_lost_counter = 0
  pt_status_msg = None
  pt_status_msg_lock = threading.Lock()


  is_scanning = False
  start_scanning = False
  is_tracking = False


  pan_tilt_errors_deg = [0.0,0.0]
  pan_tilt_goal_deg = [0.0,0.0]


  last_enabled = False

  last_img_time = nepi_utils.get_time()

  det_status_msg = None

  #######################
  ### Node Initialization
  FACTORY_NODE_NAME = "app_ai_pt_tracker" # Can be overwitten by luanch command
  def __init__(self):
    #### APP NODE INIT SETUP ####
    nepi_sdk.init_node(name= self.FACTORY_NODE_NAME)
    self.class_name = type(self).__name__
    self.base_namespace = nepi_sdk.get_base_namespace()
    self.node_name = nepi_sdk.get_node_name()
    self.node_namespace = nepi_sdk.get_node_namespace()

    ##############################  
    # Create Msg Class
    self.msg_if = MsgIF(log_name = self.class_name)
    self.msg_if.pub_info("Starting IF Initialization Processes")

    ##############################     
    # Initialize Class Variables

    self.img_data_product = self.IMAGE_DATA_PRODUCT

    ## Get folder info
    mgr_sys_srv_if = ConnectMgrSystemServicesIF()
    success = mgr_sys_srv_if.wait_for_ready()
    if success == False:
        nepi_sdk.signal_shutdown(self.node_name + ": Failed to get System Status Msg")

    #self.api_lib_folder = mgr_sys_srv_if.get_sys_folder_path('api_lib',API_LIB_FOLDER)
    #self.msg_if.pub_info("Using User Config Folder: " + str(self.api_lib_folder))

    self.api_lib_folder = API_LIB_FOLDER
    self.msg_if.pub_info("Using SDK Share Folder: " + str(self.api_lib_folder))

    ###############################
    # Launch detection img pub node that handles detection image publishing
    pkg_name = APP_PACKAGE_NAME
    node_file_folder = self.api_lib_folder
    img_pub_file = APP_IMG_PUB_FILE
    img_pub_file_path = os.path.join(node_file_folder,img_pub_file)

    if os.path.exists(img_pub_file_path) == False:
        self.msg_if.pub_warn("Could not find det img pub node file at: " + img_pub_file_path)
    else: 
        #Try and launch node
        img_pub_node_name = self.node_name + "_img_pub"
        img_pub_namespace = self.node_namespace + "_img_pub"
        self.msg_if.pub_warn("Launching Detector Img Pub Node: " + img_pub_node_name)

        # Pre Set Img Pub Params
        dp_param_ns = nepi_sdk.create_namespace(img_pub_node_name,'data_product')
        nepi_sdk.set_param(dp_param_ns,self.img_data_product)

        app_param_ns = nepi_sdk.create_namespace(img_pub_node_name,'det_namespace')
        nepi_sdk.set_param(app_param_ns,self.node_namespace)
        
        [success, msg, pub_process] = nepi_sdk.launch_node(pkg_name, img_pub_file, img_pub_node_name)

        self.msg_if.pub_warn("Detector Img Pub Node launch return msg: " + msg)



   
    # Message Image to publish when detector not running
    message = "TARGETING NOT ENABLED"
    self.app_ne_img = nepi_img.create_message_image(message)

    message = "WAITING FOR AI DETECTOR TO START"
    self.detector_nr_img = nepi_img.create_message_image(message)

    message = "WAITING FOR TARGET CLASS SELECTION"
    self.no_class_img = nepi_img.create_message_image(message)

    message = "WAITING FOR PANTILT SELECTION"
    self.no_pt_img = nepi_img.create_message_image(message)


    self.status_msg = AiPtTrackerStatus()

    self.ai_mgr_if = ConnectMgrAiModelIF()
    ready = self.ai_mgr_if.wait_for_ready()

    ##############################
    ### Setup Node

    # Configs Config Dict ####################
    self.CFGS_DICT = {
            'init_callback': self.initCb,
            'reset_callback': self.resetCb,
            'factory_reset_callback': self.factoryResetCb,
            'init_configs': True,
            'namespace': self.node_namespace
    }



    # Params Config Dict ####################
    self.PARAMS_DICT = {
        'enabled': {
            'namespace': self.node_namespace,
            'factory_val': False
        },
        'max_proc_rate_hz': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_MAX_PROC_RATE
        },
        'max_img_rate_hz': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_MAX_IMG_RATE
        },
        'scan_delay_sec': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_SCAN_UPDATE_DELAY
        },
        'selected_detector': {
            'namespace': self.node_namespace,
            'factory_val': 'None'
        },
        'image_fov_vert_degs': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_FOV_VERT_DEG
        },
        'image_fov_horz_degs': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_FOV_HORZ_DEG
        },
        'selected_class': {
            'namespace': self.node_namespace,
            'factory_val': "None"
        },
        'overlay_labels': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_LABELS_OVERLAY
        },
        'overlay_clf_name': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_CLF_OVERLAY
        },
        'overlay_img_name': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_IMG_OVERLAY
        },
        'target_queue_len': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_Q_LEN
        },
        'target_lost_len': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_L_LEN
        },
        'min_area_ratio': {
            'namespace': self.node_namespace,
            'factory_val':self.FACTORY_MIN_AREA_RATIO
        },
        'selected_pantilt': {
            'namespace': self.node_namespace,
            'factory_val': "None"
        },
        'scan_speed_ratio': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_SCAN_SPEED_RATIO
        },
        'scan_tilt_offset': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_SCAN_TILT_DEG
        },
        'min_pan_angle': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_MIN_MAX_PAN_ANGLES[0]
        },
        'max_pan_angle': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_MIN_MAX_PAN_ANGLES[1]
        },
        'min_tilt_angle': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_MIN_MAX_TILT_ANGLES[0]
        },
        'max_tilt_angle': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_MIN_MAX_TILT_ANGLES[1]
        },
        'track_speed_ratio': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TRACK_SPEED_RATIO
        },
        'track_tilt_offset': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TRACK_TILT_OFFSET_DEG
        },
        'error_goal_deg': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_ERROR_GOAL_DEG
        }
    }

    # Publishers Config Dict ####################
    self.PUBS_DICT = {
        'status_pub': {
            'namespace': self.node_namespace,
            'topic': 'status',
            'msg': AiPtTrackerStatus,
            'qsize': 1,
            'latch': True
        },
        'det_status_pub': {
            'namespace': self.node_namespace,
            'topic': 'detector_status',
            'msg': AiDetectorStatus,
            'qsize': 1,
            'latch': True
        },
        'found_object': {
            'msg': ObjectCount,
            'namespace': self.node_namespace,
            'topic': 'found_object',
            'qsize': 1,
            'latch': False
        },
        'bounding_boxes': {
            'msg': BoundingBoxes,
            'namespace': self.node_namespace,
            'topic': 'bounding_boxes',
            'qsize': 1,
            'latch': False
        },
        'errors': {
            'namespace': self.node_namespace,
            'topic': 'errors',
            'msg': TrackingErrors,
            'qsize': 1,
            'latch': True
        }
    }

    # Subscribers Config Dict ####################
    self.SUBS_DICT = {
        'publish_status': {
            'namespace': self.node_namespace,
            'topic': 'publish_status',
            'msg': Empty,
            'qsize': 10,
            'callback': self.pubStatusCb, 
            'callback_args': ()
        },
        'enable_app': {
            'namespace': self.node_namespace,
            'topic': 'enable_app',
            'msg': Bool,
            'qsize': 1,
            'callback': self.appEnableCb, 
            'callback_args': ()
        },
        'set_overlay_labels': {
            'namespace': self.node_namespace,
            'topic': 'set_overlay_labels',
            'msg': Bool,
            'qsize': 10,
            'callback': self.setOverlayLabelsCb, 
            'callback_args': ()
        },
        'set_overlay_clf_name': {
            'namespace': self.node_namespace,
            'topic': 'set_overlay_clf_name',
            'msg': Bool,
            'qsize': 10,
            'callback': self.setOverlayClfNameCb, 
            'callback_args': ()
        },
        'set_max_proc_rate': {
            'namespace': self.node_namespace,
            'topic': 'set_max_proc_rate',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setMaxProcRateCb, 
            'callback_args': ()
        },
        'set_max_img_rate': {
            'namespace': self.node_namespace,
            'topic': 'set_max_img_rate',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setMaxImgRateCb, 
            'callback_args': ()
        },
        'set_scan_delay_sec': {
            'namespace': self.node_namespace,
            'topic': 'set_scan_delay_sec',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setScanDelayCb, 
            'callback_args': ()
        },
        'set_overlay_img_name': {
            'namespace': self.node_namespace,
            'topic': 'set_overlay_img_name',
            'msg': Bool,
            'qsize': 10,
            'callback': self.setOverlayImgNameCb, 
            'callback_args': ()
        },
        'set_image_fov_vert': {
            'namespace': self.node_namespace,
            'topic': 'set_image_fov_vert',
            'msg': Float32,
            'qsize': 1,
            'callback': self.setVertFovCb, 
            'callback_args': ()
        },
        'set_image_fov_horz': {
            'namespace': self.node_namespace,
            'topic': 'set_image_fov_horz',
            'msg': Float32,
            'qsize': 1,
            'callback': self.setHorzFovCb, 
            'callback_args': ()
        },
        'select_detector': {
            'namespace': self.node_namespace,
            'topic': 'select_detector',
            'msg': String,
            'qsize': 1,
            'callback': self.setDetectorCb, 
            'callback_args': ()
        },
        'select_class': {
            'namespace': self.node_namespace,
            'topic': 'select_class',
            'msg': String,
            'qsize': 1,
            'callback': self.setClassCb, 
            'callback_args': ()
        },
        'set_target_queue_len': {
            'namespace': self.node_namespace,
            'topic': 'set_target_queue_len',
            'msg': Int32,
            'qsize': 10,
            'callback': self.setTargetQLenCb, 
            'callback_args': ()
        },
        'set_target_lost_len': {
            'namespace': self.node_namespace,
            'topic': 'set_target_lost_len',
            'msg': Int32,
            'qsize': 10,
            'callback': self.setTargetLLenCb, 
            'callback_args': ()
        },
        'set_min_area_ratio': {
            'namespace': self.node_namespace,
            'topic': 'set_min_area_ratio',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setMinAreaCb, 
            'callback_args': ()
        },
        'select_pantilt': {
            'namespace': self.node_namespace,
            'topic': 'select_pantilt',
            'msg': String,
            'qsize': 10,
            'callback': self.setPtTopicCb, 
            'callback_args': ()
        },
        'set_scan_speed_ratio': {
            'namespace': self.node_namespace,
            'topic': 'set_scan_speed_ratio',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setScanSpeedCb, 
            'callback_args': ()
        },
        'set_scan_tilt_offset': {
            'namespace': self.node_namespace,
            'topic': 'set_scan_tilt_offset',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setScanTiltOffsetCb, 
            'callback_args': ()
        },
        'set_min_max_pan_angles': {
            'namespace': self.node_namespace,
            'topic': 'set_min_max_pan_angles',
            'msg': RangeWindow,
            'qsize': 10,
            'callback': self.setMinMaxPanCb, 
            'callback_args': ()
        },
        'set_min_max_tilt_angles': {
            'namespace': self.node_namespace,
            'topic': 'set_min_max_tilt_angles',
            'msg': RangeWindow,
            'qsize': 10,
            'callback': self.setMinMaxTiltCb, 
            'callback_args': ()
        },
        'set_track_speed_ratio': {
            'namespace': self.node_namespace,
            'topic': 'set_track_speed_ratio',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setTrackSpeedCb, 
            'callback_args': ()
        },
        'set_track_tilt_offset': {
            'namespace': self.node_namespace,
            'topic': 'set_track_tilt_offset',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setTrackTiltOffsetCb, 
            'callback_args': ()
        },        
        'set_error_goal_deg': {
            'namespace': self.node_namespace,
            'topic': 'set_error_goal_deg',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setErrorGoalCb, 
            'callback_args': ()
        }          
    }


    # Create Node Class ####################
    self.node_if = NodeClassIF(
                    configs_dict = self.CFGS_DICT,
                    params_dict = self.PARAMS_DICT,
                    pubs_dict = self.PUBS_DICT,
                    subs_dict = self.SUBS_DICT
    )

    ready = self.node_if.wait_for_ready()


    # Setup Image IF
    img_namespace = nepi_sdk.create_namespace(self.node_namespace,self.img_data_product)
    self.image_if = ImageIF(namespace = img_namespace, log_name = self.img_data_product)

    time.sleep(1)

    ##############################
    self.initCb(do_updates = True)
    # Set up save data and save config services ########################################################
    factory_data_rates= {}
    for d in self.data_products:
        factory_data_rates[d] = [0.0, 0.0, 100.0] # Default to 0Hz save rate, set last save = 0.0, max rate = 100.0Hz
    if self.img_data_product in self.data_products:
        factory_data_rates[self.img_data_product] = [1.0, 0.0, 100.0] 
    self.save_data_if = SaveDataIF(data_products = self.data_products, factory_rate_dict = factory_data_rates)
    



    ##############################
    ## Start Node Processes
    self.publish_status()

    # Set up the timer that start scanning when no objects are detected
    self.msg_if.pub_info("Setting up processes")
    nepi_sdk.start_timer_process(self.UPDATER_PROCESS_DELAY, self.updaterCb, oneshot = True)

    proc_rate = self.node_if.get_param('max_proc_rate_hz')
    proc_delay = float(1) / float(proc_rate)
    nepi_sdk.start_timer_process(proc_delay, self.scanTrackCb, oneshot = True)


    ##############################
    ## Initiation Complete
    self.msg_if.pub_info(" Initialization Complete")
    # Spin forever (until object is detected)
    nepi_sdk.spin()
    ##############################

  #######################
  ### App Config Functions





  def initCb(self,do_updates = False):
    self.msg_if.pub_info(" Setting init values to param values")
    if do_updates == True:
      self.resetCb(do_updates)


  def resetCb(self,do_updates = True):
    self.publish_status()


  def factoryResetCb(self):
    self.last_image_topic = ""
    self.last_sel_pt = ""
    self.publish_status()
        



  def get_target_bearings(self,box):
      target_vert_angle_deg = 0
      target_horz_angle_deg = 0
      if self.img_height != 0 and self.img_width != 0:
        # Iterate over all of the objects and calculate range and bearing data
        image_fov_vert = self.status_msg.image_fov_vert_degs
        image_fov_horz = self.status_msg.image_fov_horz_degs
        box_y = box.ymin + (box.ymax - box.ymin)
        box_x = box.xmin + (box.xmax - box.xmin)
        box_center = [box_y,box_x]
        y_len = (box.ymax - box.ymin)
        x_len = (box.xmax - box.xmin)
        # Calculate target bearings
        object_loc_y_pix = float(box.ymin + ((box.ymax - box.ymin))  / 2) 
        object_loc_x_pix = float(box.xmin + ((box.xmax - box.xmin))  / 2)
        object_loc_y_ratio_from_center = float(object_loc_y_pix - self.img_height/2) / float(self.img_height/2)
        object_loc_x_ratio_from_center = float(object_loc_x_pix - self.img_width/2) / float(self.img_width/2)
        target_vert_angle_deg = (object_loc_y_ratio_from_center * float(image_fov_vert/2))
        target_horz_angle_deg = -1* (object_loc_x_ratio_from_center * float(image_fov_horz/2))
      return target_horz_angle_deg, target_vert_angle_deg




  ### Monitor Output of AI detector to clear detection status
  def detStatusCb(self,status_msg):
    self.det_status_msg = status_msg


  ### Monitor Output of AI detector to clear detection status
  def foundObjectCb(self,found_obj_msg):
    target_l_len = self.status_msg.target_lost_len
    #Clean Up
    if found_obj_msg.count == 0:      
      self.target_box_lock.acquire()
      self.target_box = None      
      self.target_box_lock.release()


      # Now update our target queue
      self.target_lost_counter += 1
      if self.target_lost_counter >= target_l_len:
        self.target_lost_counter = 0
        self.target_box_q_lock.acquire()
        if len(self.target_box_q) > 0:
          self.target_box_q.pop(0)
        self.target_box_q_lock.release()


  ### If object(s) detected, save bounding box info to global
  def objectDetectedCb(self,bounding_boxes_msg):
    enabled = self.status_msg.enabled
    selected_class = selected_class = self.status_msg.selected_class
    target_q_len = self.status_msg.target_queue_len
    target_l_len = self.status_msg.target_lost_len
    min_area_ratio =  self.status_msg.min_area_ratio
    get_msg_timestamp = bounding_boxes_msg.header.stamp
    bb_list = bounding_boxes_msg.bounding_boxes
    bounding_boxes_msg.bounding_boxes = [] # Clear for later use
    self.img_height = bounding_boxes_msg.image_height
    self.img_width = bounding_boxes_msg.image_width
    if enabled == False:
      self.target_box_lock.acquire()
      self.target_box = None      
      self.target_box_lock.release()
    else:
      #self.msg_if.pub_warn("Got Targets Locs: " + str(target_locs_msg))
      # Iterate over all of the objects reported by the detector and return center of largest box in degrees relative to img center
      largest_box = None
      largest_box_area_ratio=0 # Initialize largest box area
      for box in bb_list:
        # Check for the object of interest and take appropriate actions
        #self.msg_if.pub_warn("Looking for selected target: " + str(selected_class))
        if box.Class == selected_class:
          # Check if largest box
          box_area_ratio = box.area_ratio
          if box_area_ratio > largest_box_area_ratio:
            largest_box_area_ratio=box_area_ratio
            largest_box=box

      if largest_box == None or (largest_box_area_ratio < min_area_ratio and min_area_ratio != 0):

          self.target_box_lock.acquire()
          self.target_box = None      
          self.target_box_lock.release()

          self.target_lost_counter += 1
          if self.target_lost_counter >= target_l_len:
            self.target_lost_counter = 0
            self.target_box_q_lock.acquire()
            if len(self.target_box_q) > 0:
              self.target_box_q.pop(0)
            self.target_box_q_lock.release()

      else:
          self.target_lost_counter = 0
          self.target_box_lock.acquire()
          self.target_box = largest_box      
          self.target_box_lock.release()

          self.target_box_q_lock.acquire()
          if len(self.target_box_q) >= target_q_len:
           self.target_box_q.pop(0)
          self.target_box_q.append(largest_box)
          self.target_box_q_lock.release()
          bounding_boxes_msg.bounding_boxes = [largest_box]
    # Now publish our bounding boxes message
    if self.node_if is not None:
      self.node_if.publish_pub('bounding_boxes',bounding_boxes_msg)

    found_obj_msg = ObjectCount()
    found_obj_msg.header = bounding_boxes_msg.header
    found_obj_msg.model_name = bounding_boxes_msg.model_name
    found_obj_msg.image_header = bounding_boxes_msg.image_header
    found_obj_msg.image_topic = bounding_boxes_msg.image_topic
    found_obj_msg.count = len(bounding_boxes_msg.bounding_boxes)
      



  def updaterCb(self,timer):
    ############## DEBUG
    #self.node_if.set_param("enabled",True)
    #self.node_if.set_param("selected_pantilt","/nepi/s2x/iqr_pan_tilt/ptx")
    #self.node_if.set_param("selected_class","person")
    ############## DEBUG

    update_status = False
    enabled = self.node_if.get_param("enabled")

    app_msg = ""
    #self.msg_if.pub_warn("Running app update process with app enabled: " + str(enabled))
    if enabled == False:
      app_msg += "Targeting not enabled"
    elif self.last_enabled != enabled:
      update_status = True
    self.last_enabled = enabled


    # Setup PT subscribers and Publishers if needed
    sel_pt = self.node_if.get_param('selected_pantilt')
    pt_valid = sel_pt != "None" and sel_pt != ""
    pt_changed = sel_pt != self.last_sel_pt

    #self.msg_if.pub_warn("Selected PT: " + str(sel_pt))
    #self.msg_if.pub_warn("Last Selected PT: " + str(self.last_sel_pt))
    #self.msg_if.pub_warn("PT Connected: " + str(self.pt_connected))
    #self.msg_if.pub_warn("PT Valid: " + str(pt_valid))
    #self.msg_if.pub_warn("PT Sub Not None: " + str(self.pt_status_sub is not None))

    if pt_valid == False and self.pt_status_sub is not None:
      self.pt_connected = False
      self.removePtSubs()
      self.last_sel_pt = ""
      update_status = True
    if (pt_valid and self.pt_status_sub is None) or pt_changed:  
      self.setupPtSubs(sel_pt)
      self.last_sel_pt = sel_pt
      update_status = True
        
    # Check if PT still there
    if self.pt_connected == True:
      if self.pt_status_topic != "":
        pt_status_topic=nepi_sdk.find_topic(self.pt_status_topic)
        if pt_status_topic == "":
          self.msg_if.pub_warn("PT lost: " + self.pt_status_topic)
          self.pt_connected = False
          self.removePtSubs()
          update_status = True
    if self.pt_connected == True:
      app_msg += ", PanTilt connected"
    else:
      app_msg += ", PanTilt not connected"


    # Update active detectors info
    models_dict = self.ai_mgr_if.get_active_models_info_dict()
    #self.msg_if.pub_warn("Got active models dict: " + str(models_dict))
    if models_dict is not None:
      dets_list = models_dict['detector_info_list']
      #self.msg_if.pub_warn("Got detector info list: " + str(dets_list))

      dets_dict = dict()
      for info_dict in dets_list:
        #self.msg_if.pub_warn("Adding detector info dict: " + str(info_dict))
        classes_list = info_dict['classes']
        namespaces = info_dict['image_detector_namespaces']
        img_topics = info_dict['image_source_topics']
        for i2, namespace in enumerate(namespaces):
          det_dict = dict()
          det_dict['classes'] = classes_list
          det_dict['img_topic'] = img_topics[i2]
          dets_dict[namespace] = det_dict

      self.dets_dict = dets_dict
      self.detectors_list = dets_dict.keys()

      #self.msg_if.pub_warn("Got detectors dict: " + str(dets_dict))
      
      # Update selected detectors info
      cur_det = self.current_det
      #self.msg_if.pub_warn("Got current det: " + str(cur_det))
      sel_det = self.node_if.get_param('selected_detector')
      #self.msg_if.pub_warn("Got selected det: " + str(sel_det))
      #self.msg_if.pub_warn("Starting check with dets keys: " + str(self.dets_dict.keys()))
      # Update Image subscribers
      if sel_det != cur_det:
        success = self.unsubscribeDetTopic()
        time.sleep(1)
        if sel_det != "None":
          if sel_det in self.dets_dict.keys():
            self.subscribeDetTopic(sel_det)
            self.current_det = sel_det
            self.status_msg.selected_detector = sel_det
        
  
    # Check class selection
    class_sel = False
    #self.msg_if.pub_warn("sel class: " + sel_class)
    sel_class = self.node_if.get_param('selected_class')
    if len(self.classes_list) > 0:
      if sel_class  in self.classes_list:
        class_sel = True
      if class_sel == False:
        app_msg += ", Target not selected"
      else:
        app_msg += ", Target selected"
    self.class_selected = class_sel


    # Update status app msg
    self.app_msg = app_msg
    if self.app_msg != self.last_app_msg:
      update_status = True
    self.last_app_msg = app_msg
    # Publish status if needed
    if update_status == True:
      #self.msg_if.pub_info(" App update process msg: " + app_msg)
      self.publish_status()

    nepi_sdk.start_timer_process (nepi_sdk.ros_duration(self.UPDATER_PROCESS_DELAY), self.updaterCb, oneshot = True)
    
 
  def subscribeDetTopic(self,det_topic):
    self.msg_if.pub_warn("Starting det subs registration for topic: " + str(det_topic))
    #if self.det_if is not None:
      #ret = self.unsubscribeDetTopic()
    if det_topic in self.dets_dict.keys():
      det_dict = self.dets_dict[det_topic]
      self.classes_list = det_dict['classes']
      self.img_source_topic = det_dict['img_topic']

      #self.msg_if.pub_warn("Registering with det dict: " + str(det_dict))
      #self.msg_if.pub_warn("Registering classes to: " + str(det_dict['classes']))

      ####################
      # Pubs Config Dict 
      SUBS_DICT = {
          'status_sub': {
              'msg': AiDetectorStatus,
              'namespace': det_topic,
              'topic': 'detector_status',
              'qsize': 10,
              'callback': self.detStatusCb, 
              'callback_args': ()
          },
          'found_object': {
              'msg': ObjectCount,
              'namespace': det_topic,
              'topic': 'found_object',
              'qsize': 10,
              'callback': self.foundObjectCb, 
              'callback_args': ()
          },
          'bounding_boxes': {
              'msg': BoundingBoxes,
              'namespace': det_topic,
              'topic': 'bounding_boxes',
              'qsize': 10,
              'callback': self.objectDetectedCb, 
              'callback_args': ()
          }
      }

      ####################
      # Subs Config Dict 

      self.det_if = NodeSubscribersIF(
                      subs_dict = SUBS_DICT,
                      log_class_name = False
      )
      self.current_det = det_topic
      self.detector_connected = True
      self.msg_if.pub_warn('Registered subs with for detector: ' + det_topic)

    return True
                

  def unsubscribeDetTopic(self):
    if self.det_if is not None:
        self.msg_if.pub_warn('Unsubscribing det subs for detector: ' + self.current_det)
        self.detector_connected = False
        self.det_if.unregister_subs()
        self.det_if = None
        self.classes_list = []
        self.current_det = "None"
        self.img_source_topic = "None"
        self.det_status_msg = "None"
        
    return True


  def setupPtSubs(self,selected_pantilt):
    if selected_pantilt is not None:
      if selected_pantilt != "None" and selected_pantilt != "":
        pt_status_topic = os.path.join(selected_pantilt,"/ptx/status")
        #self.msg_if.pub_info("Looking for topic name: " + pt_status_topic)
        self.pt_status_topic=nepi_sdk.find_topic(pt_status_topic)
        if self.pt_status_sub is not None:
          self.pt_connected = False
          self.removePtSubs()
        if self.pt_status_topic == "":
          self.selected_pantilt = "None"
        if self.pt_status_topic != "":
          self.msg_if.pub_info("Found ptx status topic: " + self.pt_status_topic)
          ptx_namespace = self.pt_status_topic.replace("status","")
          self.msg_if.pub_info("Found ptx namespace: " + ptx_namespace)
          selected_pantilt = ptx_namespace.split("/ptx")[0]
          self.selected_pantilt = nepi_sdk.crate_namespace(selected_pantilt,"/ptx")
          # PanTilt Status Topics
          # PanTilt Control Publish Topics
          PTX_SET_SPEED_RATIO_TOPIC = ptx_namespace + "set_speed_ratio"
          PTX_GOHOME_TOPIC = ptx_namespace + "go_home"
          PTX_STOP_TOPIC = ptx_namespace + "stop_moving"
          PTX_GOTO_PAN_RATIO_TOPIC = ptx_namespace + "jog_to_yaw_ratio"
          PTX_GOTO_TILT_RATIO_TOPIC = ptx_namespace + "jog_to_pitch_ratio"
          PTX_JOG_PAN_TOPIC = ptx_namespace + "jog_timed_yaw"
          PTX_JOG_TILT_TOPIC = ptx_namespace + "jog_timed_pitch"
          PTX_JOG_POSITION_TOPIC = ptx_namespace + "jog_to_position"
          PTX_SET_SOFT_LIMITS_TOPIC = ptx_namespace + "set_soft_limits"

          ## Get PTX capabilities info

          ptx_capabilities_service_topic = ptx_namespace + "capabilities_query"
          try:
            ptx_caps_service = nepi_sdk.connect_service(ptx_capabilities_service_topic, PTXCapabilitiesQuery)
            time.sleep(1)
            ptx_caps = ptx_caps_service()
            self.has_position_feedback = ptx_caps.has_absolute_positioning
            self.has_adjustable_speed =  ptx_caps.has_adjustable_speed
            self.has_auto_pan =  ptx_caps.has_auto_pan
            self.has_auto_tilt =  ptx_caps.has_auto_tilt
          except Exception as e:
            self.msg_if.pub_warn("Failed to call PTX capabilities service: " + ptx_capabilities_service_topic + " " + str(e))
            self.has_position_feedback = False
            self.has_adjustable_speed =  False
          '''
          ## Create Publishers
          self.send_pt_home_pub = nepi_sdk.create_publisher(PTX_GOHOME_TOPIC, Empty, queue_size=10)
          self.set_pt_speed_ratio_pub = nepi_sdk.create_publisher(PTX_SET_SPEED_RATIO_TOPIC, Float32, queue_size=10)
          self.set_pt_position_pub = nepi_sdk.create_publisher(PTX_JOG_POSITION_TOPIC, PanTiltPosition, queue_size=10)
          self.set_pt_pan_ratio_pub = nepi_sdk.create_publisher(PTX_GOTO_PAN_RATIO_TOPIC, Float32, queue_size=10)
          self.set_pt_tilt_ratio_pub = nepi_sdk.create_publisher(PTX_GOTO_TILT_RATIO_TOPIC, Float32, queue_size=10)
          self.set_pt_pan_jog_pub = nepi_sdk.create_publisher(PTX_JOG_PAN_TOPIC, SingleAxisTimedMove, queue_size=10)
          self.set_pt_tilt_jog_pub = nepi_sdk.create_publisher(PTX_JOG_TILT_TOPIC, SingleAxisTimedMove, queue_size=10)
          self.set_pt_soft_limits_pub = nepi_sdk.create_publisher(PTX_SET_SOFT_LIMITS_TOPIC, PanTiltLimits, queue_size=10)
          self.pt_stop_motion_pub = nepi_sdk.create_publisher(PTX_STOP_TOPIC, Empty, queue_size=10)
          time.sleep(1)
          ## Create Subscribers
          self.msg_if.pub_info("Subscribing to PTX Status Msg: " + self.pt_status_topic)
          self.pt_status_sub = nepi_sdk.create_subscriber(self.pt_status_topic, PTXStatus, self.ptStatusCb, queue_size = 1)
          #self.pt_connected = True # Set in pt_status callback
      else:
        self.selected_pantilt = "None"
        self.pt_connected = False
        self.pan_tilt_errors_deg = [0,0]
    else:
      self.selected_pantilt = "None"
      self.pt_connected = False
      self.pan_tilt_errors_deg = [0,0]
      

  def removePtSubs(self):
    if self.pt_status_sub is not None:
      ## Create Class Subscribers
      self.msg_if.pub_warn("Unsubscribing to PTX Status Msg: " + self.pt_status_topic)
      self.pt_status_sub.unregister()
      self.send_pt_home_pub.unregister()
      self.set_pt_speed_ratio_pub.unregister()
      self.set_pt_position_pub.unregister()
      self.set_pt_pan_ratio_pub.unregister()
      self.set_pt_tilt_ratio_pub.unregister()
      self.set_pt_pan_jog_pub.unregister()
      self.set_pt_tilt_jog_pub.unregister()
      self.set_pt_soft_limits_pub.unregister()
      self.pt_stop_motion_pub.unregister()

      time.sleep(1)

      self.pt_status_sub = None
      self.send_pt_home_pub = None
      self.set_pt_speed_ratio_pub = None
      self.set_pt_position_pub = None
      self.set_pt_pan_ratio_pub = None
      self.set_pt_tilt_ratio_pub = None
      self.set_pt_pan_jog_pub = None
      self.set_pt_tilt_jog_pub = None
      self.set_pt_soft_limits_pub = None
      self.pt_stop_motion_pub = None


    self.has_position_feedback = False
    self.has_adjustable_speed =  False

    self.selected_pantilt = ""
    self.pt_connected = False
    self.pan_tilt_errors_deg = [0,0]

  #######################
  ### Node Callbacks

  def pubStatusCb(self,msg):
    self.publish_status()

  def appEnableCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    self.status_msg.enabled = val
    self.publish_status(do_updates = False) # Updated Here
    self.node_if.set_param('enabled',val)



  def setDetectorCb(self,msg):
    ##self.msg_if.pub_info(msg)
    selected_det = msg.data
    if selected_det in self.detectors_list or selected_det == "None":
      self.status_msg.selected_detector =  selected_det
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('selected_detector',  selected_det)

    

  def setClassCb(self,msg):
    ##self.msg_if.pub_info(msg)
    selected_class = msg.data
    if selected_class in self.classes_list or selected_class == "None":
      self.status_msg.selected_class = selected_class
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('selected_class',  selected_class)

    
    

  def setPtTopicCb(self,msg):
    ##self.msg_if.pub_info(msg)
    pt_topic = msg.data
    if pt_topic == "None":
      pt_topic = ""
    if pt_topic != "":
      pt_topic = nepi_sdk.find_topic(pt_topic)
    self.status_msg.selected_pantilt = pt_topic
    self.publish_status(do_updates = False) # Updated Here
    self.node_if.set_param('selected_pantilt',  pt_topic)

    


  def setVertFovCb(self,msg):
    ##self.msg_if.pub_info(msg)
    fov = msg.data
    if fov > 0:
      self.status_msg.image_fov_vert_degs = fov
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('image_fov_vert_degs',  fov)



  def setHorzFovCb(self,msg):
    ##self.msg_if.pub_info(msg)
    fov = msg.data
    if fov > 0:
      self.status_msg.image_fov_horz_degs = fov
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('image_fov_horz_degs',  fov)



  def setTargetQLenCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val < 1:
      val = 1
    if val > 20:
      val = 20
    self.status_msg.target_q_len = val
    self.publish_status(do_updates = False) # Updated Here
    self.node_if.set_param('target_queue_len',  val)


  def setTargetLLenCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val < 1:
      val = 1
    if val > 20:
      val = 20
    self.status_msg.target_l_len = val
    self.publish_status(do_updates = False) # Updated Here
    self.node_if.set_param('target_lost_len',  val)



  def setMinAreaCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.status_msg.min_area_ratio =val
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('min_area_ratio',  val)


  def setScanSpeedCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.status_msg.scan_speed_ratio = val
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('scan_speed_ratio',  val)


  def setScanTiltOffsetCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data

    min_tilt = self.node_if.get_param("min_tilt_angle")
    max_tilt = self.node_if.get_param("max_tilt_angle")
    if val >= min_tilt and val <= max_tilt:
      self.status_msg.scan_tilt_offset = val
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('scan_tilt_offset',  val)


  def setMinMaxPanCb(self,msg):
    min_pan = msg.start_range
    max_pan = msg.stop_range
    ##self.msg_if.pub_info(msg)
    if min_pan >= -180 and max_pan <= 180 and min_pan < max_pan:
      self.status_msg.set_pan_min_max_deg = [min_pan,max_pan]
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param("min_pan_angle",min_pan)
      self.node_if.set_param("max_pan_angle",max_pan)


  def setMinMaxTiltCb(self,msg):
    min_tilt = msg.start_range
    max_tilt = msg.stop_range
    ##self.msg_if.pub_info(msg)
    if min_tilt >= -180 and max_tilt <= 180 and min_tilt < max_tilt:
      self.status_msg.set_tilt_min_max_deg = [min_tilt,max_tilt]
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param("min_tilt_angle",min_tilt)
      self.node_if.set_param("max_tilt_angle",max_tilt)



  def setTrackSpeedCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.status_msg.track_speed_ratio = val
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('track_speed_ratio',  val)

  def setTrackTiltOffsetCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    min_pan = self.status_msg.set_pan_min_max_deg[0]
    max_pan = self.status_msg.set_pan_min_max_deg[1]
    min_tilt = self.status_msg.set_tilt_min_max_deg[0]
    max_tilt = self.status_msg.set_tilt_min_max_deg[1]
    if val >= min_tilt and val <= max_tilt:
      self.status_msg.track_tilt_offset = val
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('track_tilt_offset',  val)


  def setErrorGoalCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val < self.MIN_MAX_ERROR_GOAL[0]:
      val = self.MIN_MAX_ERROR_GOAL[0]
    if val > self.MIN_MAX_ERROR_GOAL[1]:
      val = self.MIN_MAX_ERROR_GOAL[1]
    self.status_msg.error_goal = val
    self.publish_status(do_updates = False) # Updated Here
    self.node_if.set_param('error_goal_deg',  val)



  def setOverlayLabelsCb(self,msg):
      self.status_msg.overlay_labels = msg.data
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('overlay_labels', msg.data)


  def setOverlayClfNameCb(self,msg):
      self.status_msg.overlay_clf_name = msg.data
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('overlay_clf_name', msg.data)


  def setOverlayImgNameCb(self,msg):
      self.status_msg.overlay_img_name = msg.data
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('overlay_img_name', msg.data)


  def setMaxProcRateCb(self,msg):
      max_rate = msg.data
      if max_rate <  self.MIN_MAX_RATE:
          max_rate = self.MIN_MAX_RATE
      elif max_rate > MAX_MAX_RATE:
          max_rate = MAX_MAX_RATE
      self.status_msg.max_proc_rate_hz = max_rate
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('max_proc_rate_hz', max_rate)


  def setMaxImgRateCb(self,msg):
      max_rate = msg.data
      if max_rate <  self.MIN_MAX_RATE:
          max_rate = self.MIN_MAX_RATE
      elif max_rate > MAX_MAX_RATE:
          max_rate = MAX_MAX_RATE
      self.status_msg.max_img_rate_hz = max_rate
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('max_img_rate_hz', max_rate)


  def setScanDelayCb(self,msg):
      scan_delay = msg.data
      if scan_delay < 0.01:
          scan_delay = 0.01
      elif scan_delay > 1:
          scan_delay = 1
      self.status_msg.scan_delay_sec = scan_delay
      self.publish_status(do_updates = False) # Updated Here
      self.node_if.set_param('scan_delay_sec', scan_delay)



  #######################
  ### PT Callbacks


  ### Simple callback to get pt pt_status_msg info
  def ptStatusCb(self,pt_status_msg):
    min_pan = self.status_msg.set_pan_min_max_deg[0]
    max_pan = self.status_msg.set_pan_min_max_deg[1]
    min_tilt = self.status_msg.set_tilt_min_max_deg[0]
    max_tilt = self.status_msg.set_tilt_min_max_deg[1]
    # This is just to get the current pt positions
    if pt_status_msg.yaw_now_deg != self.last_pan_pos:
      self.last_pan_pos = pt_status_msg.yaw_now_deg
      self.is_moving = True
    else:
      self.is_moving = False

    pan_min = pt_status_msg.yaw_min_softstop_deg
    pan_max = pt_status_msg.yaw_max_softstop_deg
    if min_pan < pan_min:
      min_pan = pan_min
    if max_pan > pan_max:
      max_pan = pan_max
    self.node_if.set_param("min_pan_angle",min_pan)
    self.node_if.set_param("max_pan_angle",max_pan)
    

    tilt_min = pt_status_msg.pitch_min_softstop_deg
    tilt_max = pt_status_msg.pitch_max_softstop_deg
    if min_tilt < tilt_min:
      min_tilt = tilt_min
    if max_tilt > tilt_max:
      max_tilt = tilt_max
    self.node_if.set_param("min_tilt_angle",min_tilt)
    self.node_if.set_param("max_tilt_angle",max_tilt)

    self.has_position_feedback = pt_status_msg.has_position_feedback
    self.has_adjustable_speed =  pt_status_msg.has_adjustable_speed
    self.has_auto_pan = self.has_adjustable_speed
    self.cur_speed_ratio = pt_status_msg.speed_ratio
    self.pt_status_msg_lock.acquire()
    self.pt_status_msg = pt_status_msg
    self.pt_status_msg_lock.release()

    self.pt_connected = True
  
  
  def scanTrackCb(self,timer):
    start_time = nepi_utils.get_time()
    self.pt_status_msg_lock.acquire()
    pt_status_msg = copy.deepcopy(self.pt_status_msg)
    self.pt_status_msg_lock.release()    
    was_tracking = copy.deepcopy(self.is_tracking)
    was_scanning = copy.deepcopy(self.is_scanning)
    enabled = self.status_msg.enabled
    if enabled == False or self.pt_connected == False or pt_status_msg is None:
      #self.msg_if.pub_warn("Scan Track process not ready:" + str([enabled,self.pt_connected,pt_status_msg]))
      self.target_detected=False
      self.pan_tilt_errors_deg = [0,0]
      self.is_tracking = False
      self.is_scanning = False
      if was_tracking or was_scanning:
        self.publish_status()
    else:
      #self.msg_if.pub_warn("Scan Track process ready")  
      self.target_box_q_lock.acquire()
      box_q = copy.deepcopy(self.target_box_q)      
      self.target_box_q_lock.release()



      min_pan = self.status_msg.set_pan_min_max_deg[0]
      max_pan = self.status_msg.set_pan_min_max_deg[1]
      min_tilt = self.status_msg.set_tilt_min_max_deg[0]
      max_tilt = self.status_msg.set_tilt_min_max_deg[1]


      tilt_cur = pt_status_msg.pitch_now_deg
      tilt_goal = pt_status_msg.pitch_goal_deg

      pan_cur = pt_status_msg.yaw_now_deg
      pan_goal = pt_status_msg.yaw_goal_deg    
      self.msg_if.pub_warn("Got pan tilt current: " + str([round(pan_cur,3),round(tilt_cur,3)]))
        
      #self.msg_if.pub_warn("Length of box queue: " + str(len(box_q)))
      if len(box_q) == 0: ### Scanning if no targets queued
        self.msg_if.pub_warn("Starting Scan Process") 
        self.target_detected=False
        self.pan_tilt_errors_deg = [0,0]
        self.is_tracking = False
        self.is_scanning = True
        if was_scanning == False:
          self.start_scanning = True
          self.publish_status()
        scan_speed_ratio = self.status_msg.scan_speed_ratio
        scan_tilt_offset = self.status_msg.scan_tilt_offset
        
        '''
        # Check tilt limits
        if scan_tilt_offset < min_tilt:
          scan_tilt_offset = min_tilt
        if scan_tilt_offset > max_tilt:
          scan_tilt_offset = max_tilt
        self.node_if.set_param("scan_tilt_offset",scan_tilt_offset)


        if self.has_adjustable_speed == True and self.cur_speed_ratio != scan_speed_ratio:
          try:
            self.set_pt_speed_ratio_pub.publish(scan_speed_ratio)
          except:
            pass
        '''
        if self.has_position_feedback == True:
          if (pan_cur < (min_pan + 10)):
            pan_tilt_pos_msg = PanTiltPosition()
            pan_tilt_pos_msg.yaw_deg = max_pan
            pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
            self.msg_if.pub_warn("Scanning to pan tilt : " + str(pan_tilt_pos_msg))
            self.last_scan_goal = pan_tilt_pos_msg
            try:
             self.set_pt_position_pub.publish(pan_tilt_pos_msg)
             self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
             self.current_scan_dir = 1
             self.publish_status(do_updates = False)
            except Exception as e:
              self.msg_if.pub_warn("Scanning to max_pan excpetion: " + str(e))
            #self.msg_if.pub_warn("Scanning to max_pan")
            self.start_scanning = False

          elif (pan_cur > (max_pan - 10)):
            pan_tilt_pos_msg = PanTiltPosition()
            pan_tilt_pos_msg.yaw_deg = min_pan
            pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
            self.msg_if.pub_warn("Scanning to pan tilt : " + str(pan_tilt_pos_msg))
            self.last_scan_goal = pan_tilt_pos_msg
            try:
              self.set_pt_position_pub.publish(pan_tilt_pos_msg)
              self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
              self.current_scan_dir = -1
              self.publish_status(do_updates = False)
            except:
              self.msg_if.pub_warn("Scanning to min_pan excpetion: " + str(e))
            #self.msg_if.pub_warn("Scanning to min_pan")
            self.start_scanning = False

          elif self.start_scanning == True:
            if self.current_scan_dir > 0:
              pan_tilt_pos_msg = PanTiltPosition()
              pan_tilt_pos_msg.yaw_deg = max_pan
              pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
              self.msg_if.pub_warn("Starting Scan to pan tilt : " + str(pan_tilt_pos_msg))
              self.last_scan_goal = pan_tilt_pos_msg
              try:
                self.set_pt_position_pub.publish(pan_tilt_pos_msg)
                self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
                self.publish_status(do_updates = False)
              except:
                self.msg_if.pub_warn("Starting Scan to max_pan excpetion: " + str(e))
              #self.msg_if.pub_warn("Starting to max_pan")
              self.start_scanning = False
            else:
              pan_tilt_pos_msg = PanTiltPosition()
              pan_tilt_pos_msg.yaw_deg = min_pan
              pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
              self.msg_if.pub_warn("Starting Scan to pan tilt : " + str(pan_tilt_pos_msg))
              self.last_scan_goal = pan_tilt_pos_msg
              try:
                self.set_pt_position_pub.publish(pan_tilt_pos_msg)
                self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
                self.publish_status(do_updates = False)
              except:
                self.msg_if.pub_warn("Starting Scan to min_pan excpetion: " + str(e))
              self.msg_if.pub_warn("Starting to min_pan")
              self.start_scanning = False
          else:
            self.msg_if.pub_warn("Cont Scan to pan tilt : " + str(self.last_scan_goal))
            pan_tilt_pos_msg = self.last_scan_goal
            pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
            try:
             self.set_pt_position_pub.publish(pan_tilt_pos_msg)
             self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
             self.current_scan_dir = 1
             self.publish_status(do_updates = False)
            except Exception as e:
              self.msg_if.pub_warn("Cont to max_pan excpetion: " + str(e))
        self.msg_if.pub_warn("Ending Scan Process") 

      else:   # Run Tracking
        self.target_detected=True
        self.is_tracking = True
        self.is_scanning = False
        self.start_scanning = True


        self.last_track_time = nepi_utils.get_time() 
        error_goal = self.node_if.get_param('error_goal_deg')
        track_speed_ratio = self.status_msg.track_speed_ratio
        track_tilt_offset = self.status_msg.track_tilt_offset    
        '''
        if self.has_adjustable_speed == True and self.cur_speed_ratio != track_speed_ratio:
          try:
            self.set_pt_speed_ratio_pub.publish(track_speed_ratio)
          except:
            pass
        '''
        self.msg_if.pub_warn("Error Goal set to: " + str(error_goal))
        self.msg_if.pub_warn("Got Targets Errors pan tilt: " + str(self.pan_tilt_errors_deg))
        pan_error_q = []
        tilt_error_q = []
        for box in box_q:
          [pan_error,tilt_error] = self.get_target_bearings(box)
          pan_error_q.append(pan_error)
          tilt_error_q.append(tilt_error)
        pan_error_avg = sum(pan_error_q) / len(pan_error_q)
        tilt_error_avg = sum(tilt_error_q) / len(tilt_error_q) + track_tilt_offset
  

        pan_to_goal = pan_cur + pan_error_avg /2
        tilt_to_goal = tilt_cur + tilt_error_avg /2 

        self.pan_tilt_errors_deg = [pan_error_avg,tilt_error_avg]


        if pan_error_avg > 0:
            self.last_track_dir = 1
        else: 
            self.last_track_dir = -1
        self.current_scan_dir = self.last_track_dir

        if abs(tilt_error_avg) < error_goal and abs(pan_error_avg) < error_goal:
            #self.msg_if.pub_warn("Tracking within error bounds")
            #try:
              #self.pt_stop_motion_pub.publish(Empty())
            #except:
              #pass
            pass
        else:
            # Set pan angle goal

            if abs(pan_error_avg) < error_goal:
                pan_to_goal = pan_cur
            if pan_to_goal < min_pan:
                pan_to_goal = min_pan
            if pan_to_goal > max_pan:
                pan_to_goal = max_pan

            # Set tilt angle goal

            if abs(tilt_error_avg) < error_goal:
                tilt_to_goal = tilt_cur
            if tilt_to_goal < min_tilt:
                tilt_to_goal = min_tilt
            if tilt_to_goal > max_tilt:
                tilt_to_goal = max_tilt

            self.msg_if.pub_warn("Current Pos: " + str([pan_cur,tilt_cur]))
            self.msg_if.pub_warn("Track to Pos: " + str([pan_to_goal,tilt_to_goal]))
            if self.has_position_feedback == True:
              # Send angle goal
              pt_pos_msg = PanTiltPosition()
              pt_pos_msg.yaw_deg = pan_to_goal
              pt_pos_msg.pitch_deg = tilt_to_goal
              if not nepi_sdk.is_shutdown():
                try:
                  self.set_pt_position_pub.publish(pt_pos_msg)   
                  self.pan_tilt_goal_deg = [pan_to_goal,tilt_to_goal]
                except:
                  self.msg_if.pub_warn("Tracking to excpetion: " + str(e))
                self.msg_if.pub_warn("Tracking to pan tilt: " + str(pan_to_goal) + " " + str(tilt_to_goal))
              else:
                pass # add timed jog controls

        if was_tracking == False:
          self.publish_status(do_updates = False)
    # Publish errors msg
    if pt_status_msg is not None:
      tracking_error_msg = TrackingErrors()
      tracking_error_msg.yaw_cur_deg = pt_status_msg.yaw_now_deg
      tracking_error_msg.yaw_error_deg = self.pan_tilt_errors_deg[0]
      tracking_error_msg.yaw_goal_deg = self.pan_tilt_goal_deg[0]
      tracking_error_msg.pitch_cur_deg = pt_status_msg.pitch_now_deg
      tracking_error_msg.pitch_error_deg = self.pan_tilt_errors_deg[1]
      tracking_error_msg.pitch_goal_deg = self.pan_tilt_goal_deg[1]
      if self.node_if is not None:
        self.node_if.publish_pub('errors', tracking_error_msg)

    proc_time = nepi_utils.get_time() - start_time
    if self.is_scanning == True:
      proc_delay = self.status_msg.scan_delay_sec
    else:
      proc_rate = self.status_msg.max_proc_rate_hz
      proc_delay = float(1) / float(proc_rate)
    if proc_delay > proc_time:
      proc_delay = proc_delay - proc_time

    if self.is_scanning == True:
      proc_delay = 2
    else:
      proc_rate = 2


    nepi_sdk.start_timer_process(proc_delay, self.scanTrackCb, oneshot = True)



    ###################
    ## Detector Status Publish


    def create_classes_colors_msg(self,classes_list):
        colors_msg_list = []
        for class_name in classes_list:
            color_msg  = ColorRGBA()
            class_ind = classes_list.index(class_name)
            if len(self.classes_colors) >= class_ind:
                color = self.classes_colors[class_ind]
                color_msg.r = color[0]
                color_msg.g = color[1]
                color_msg.b = color[2]
            else:
                color_msg.r = 0
                color_msg.g = 200
                color_msg.b = 0
            colors_msg_list.append(color_msg)
        return colors_msg_list


    def publish_det_status(self, do_updates = True):
        if self.det_status_msg is None:
          status_msg = AiDetectorStatus()
          status_msg.name = self.model_name



        else:
          status_msg = self.det_status_msg

        status_msg.name = self.model_name

        status_msg.namespace = self.node_namespace
        status_msg.state = self.state


        if do_updates == True:
            status_msg.enabled = self.node_if.get_param('enabled')
            sel_classes = self.node_if.get_param('selected_classes')
            status_msg.selected_classes = sel_classes
            status_msg.selected_classes_colors = self.create_classes_colors_msg(sel_classes)
               

            status_msg.sleep_enabled = self.node_if.get_param('sleep_enabled')
            status_msg.sleep_suspend_sec = self.node_if.get_param('sleep_suspend_sec')
            status_msg.sleep_run_sec = self.node_if.get_param('sleep_run_sec')
            status_msg.sleep_state = self.sleep_state

            status_msg.img_tiling = self.node_if.get_param('img_tiling')

            status_msg.overlay_labels = self.node_if.get_param('overlay_labels')
            status_msg.overlay_clf_name = self.node_if.get_param('overlay_clf_name')
            status_msg.overlay_img_name = self.node_if.get_param('overlay_img_name')

            status_msg.threshold = self.node_if.get_param('threshold')
            status_msg.max_proc_rate_hz = self.node_if.get_param('max_proc_rate_hz')
            status_msg.max_img_rate_hz = self.node_if.get_param('max_img_rate_hz')


            status_msg.selected_img_topics = self.node_if.get_param('selected_img_topics')

        img_source_topics = []
        img_det_namespaces = []
        img_det_states = []
        img_connects = []
        img_img_lat_times = []
        img_det_lat_times = []
        img_pre_times = []
        img_detect_times = []
        imgs_info_dict = copy.deepcopy(self.imgs_info_dict)
        for img_topic in imgs_info_dict.keys():
                state = imgs_info_dict[img_topic]['active']
                if state == True:
                    img_source_topics.append(img_topic)
                    img_det_namespaces.append(imgs_info_dict[img_topic]['namespace'])
                    img_connects.append(imgs_info_dict[img_topic]['connected'])
                    img_img_lat_times.append(imgs_info_dict[img_topic]['image_latency_time'])
                    img_det_lat_times.append(imgs_info_dict[img_topic]['detect_latency_time'])
                    img_pre_times.append(imgs_info_dict[img_topic]['preprocess_time'])
                    img_detect_times.append(imgs_info_dict[img_topic]['detect_time'])
        status_msg.image_source_topics = img_source_topics
        status_msg.image_detector_namespaces = img_det_namespaces
        status_msg.images_connected = img_connects
        status_msg.image_latency_times = img_img_lat_times
        status_msg.detect_latency_times = img_det_lat_times
        status_msg.preprocess_times = img_pre_times
        status_msg.detect_times = img_detect_times




        #self.msg_if.pub_warn("Sending Status Msg: " + str(status_msg))
        if self.node_if is not None:
          self.node_if.publish_pub('det_status_pub',status_msg)



  ###################
  ## Status Publisher
  def publish_status(self, do_updates = True):

    if do_updates == True:

      self.status_msg.enabled = self.node_if.get_param('enabled')


      self.status_msg.max_proc_rate_hz = self.node_if.get_param('max_proc_rate_hz')
      self.status_msg.max_img_rate_hz = self.node_if.get_param('max_img_rate_hz')
      
      self.status_msg.image_fov_vert_degs = self.node_if.get_param('image_fov_vert_degs')
      self.status_msg.image_fov_horz_degs = self.node_if.get_param('image_fov_horz_degs')


      min_pan = self.node_if.get_param("min_pan_angle")
      max_pan = self.node_if.get_param("max_pan_angle")
      self.status_msg.set_pan_min_max_deg = [min_pan,max_pan]

      min_tilt = self.node_if.get_param("min_tilt_angle")
      max_tilt = self.node_if.get_param("max_tilt_angle")
      self.status_msg.set_tilt_min_max_deg = [min_tilt,max_tilt]
      

      self.status_msg.min_area_ratio = self.node_if.get_param("min_area_ratio")

      self.status_msg.scan_delay_sec = self.node_if.get_param("scan_delay_sec")
      self.status_msg.scan_speed_ratio = self.node_if.get_param("scan_speed_ratio")
      self.status_msg.scan_tilt_offset = self.node_if.get_param("scan_tilt_offset")

      self.status_msg.track_speed_ratio = self.node_if.get_param("track_speed_ratio")
      self.status_msg.track_tilt_offset = self.node_if.get_param("track_tilt_offset")

      self.status_msg.error_goal_min_max_deg = self.MIN_MAX_ERROR_GOAL
      self.status_msg.error_goal_deg = self.node_if.get_param('error_goal_deg')


      self.status_msg.target_queue_len = self.node_if.get_param('target_queue_len')
      self.status_msg.target_lost_len = self.node_if.get_param('target_lost_len')



    self.status_msg.available_detectors = sorted(self.detectors_list)
    selected_detector = self.node_if.get_param('selected_detector')
    if selected_detector not in self.detectors_list:
      selected_detector = "None"
    self.status_msg.selected_detector = selected_detector 


    self.status_msg.image_source_topic = self.img_source_topic


    self.status_msg.available_classes = sorted(self.classes_list)
    selected_class = self.node_if.get_param('selected_class')
    if selected_class not in self.classes_list:
      selected_class = "None"
    self.status_msg.selected_class = selected_class 
    self.status_msg.target_detected = self.target_detected

    self.status_msg.selected_pantilt = self.selected_pantilt
    self.status_msg.pantilt_connected = self.pt_connected
    self.status_msg.has_position_feedback = self.has_position_feedback
    self.status_msg.has_adjustable_speed = self.has_adjustable_speed
    self.status_msg.has_auto_pan = self.has_auto_pan
    self.status_msg.has_auto_tilt = self.has_auto_tilt
    self.pt_status_msg_lock.acquire()
    pt_status_msg = copy.deepcopy(self.pt_status_msg)
    self.pt_status_msg_lock.release()
    if pt_status_msg is not None:
      pan_min = pt_status_msg.yaw_min_softstop_deg
      pan_max = pt_status_msg.yaw_max_softstop_deg
      self.status_msg.pan_min_max_deg = [pan_min,pan_max]

      tilt_min = pt_status_msg.pitch_min_softstop_deg
      tilt_max = pt_status_msg.pitch_max_softstop_deg
      self.status_msg.tilt_min_max_deg = [tilt_min,tilt_max]
    else:
      self.status_msg.pan_min_max_deg = [-180,180]
      self.status_msg.tilt_min_max_deg = [-180,180]


    self.status_msg.msg = self.app_msg


    self.target_box_q_lock.acquire()
    box_q = copy.deepcopy(self.target_box_q)      
    self.target_box_q_lock.release()
    self.status_msg.target_queue_count = len(box_q)

    self.status_msg.is_scanning = self.is_scanning
    self.status_msg.is_tracking = self.is_tracking
    self.status_msg.pan_direction = self.current_scan_dir

    #self.msg_if.pub_info("Printing status: " + str(self.status_msg))
    if self.node_if is not None:
      self.node_if.publish_pub('status_pub', self.status_msg)





  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    global led_intensity_pub
    self.msg_if.pub_info("Shutting down: Executing script cleanup actions")




#########################################
# Main
#########################################
if __name__ == '__main__':
  pantiltTargetTrackerApp()


