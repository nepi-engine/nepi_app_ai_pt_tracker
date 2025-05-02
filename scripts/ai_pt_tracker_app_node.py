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

from nepi_sdk import nepi_ros 
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_img

from std_msgs.msg import Bool, UInt8, Empty, Int32,Float32, String
from sensor_msgs.msg import Image

from nepi_ros_interfaces.msg import PanTiltLimits, PanTiltPosition, SingleAxisTimedMove, PTXStatus, StringArray
from nepi_ros_interfaces.srv import PTXCapabilitiesQuery

from nepi_ros_interfaces.msg import BoundingBox, BoundingBoxes, ObjectCount, RangeWindow
from nepi_ros_interfaces.msg import AiModelInfo, AiModelMgrStatus,

from nepi_app_ai_targeting.msg import AiTargetingStatus
from nepi_app_ai_pt_tracker.msg import AiPtTrackerStatus , TrackingErrors


from nepi_api.node_if import NodeClassIF
from nepi_api.connect_node_if import ConnectNodeClassIF
from nepi_api.messages_if import MsgIF
from nepi_api.system_if import SaveDataIF
from nepi_api.system_if import SaveCfgIF
from nepi_api.data_if import ImageIF


#########################################
# Node Class
#########################################

class pantiltTargetTrackerApp(object):
  AI_MANAGER_NODE_NAME = "ai_detector_mgr"

  UPDATER_PROCESS_DELAY = 1
  SCAN_TRACK_PROCESS_DELAY = 1
  IMG_PUB_PROCESS_DELAY = 0.2

  PTX_MAX_TRACK_SPEED_RATIO = 1.0
  PTX_MIN_TRACK_SPEED_RATIO = 0.1
  PTX_OBJ_CENTERED_BUFFER_RATIO = 0.15 # Hysteresis band about center of image for tracking purposes


# Pan and Tilt tracking settings

  FACTORY_FOV_VERT_DEG=70 # Camera Vertical Field of View (FOV)
  FACTORY_FOV_HORZ_DEG=110 # Camera Horizontal Field of View (FOV)


  FACTORY_TRACK_UPDATE_RATE = 1
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

  MIN_TRACK_UPDATE_RATE = 1
  MAX_TRACK_UPDATE_RATE = 10
  TRACK_JOG_TIME = 0.5

  data_products = ["tracking_image"]
  targeting_status_msg = None


  ai_mgr_status_msg = None

  detectors_list = []
  last_detector = ""

  current_image_topic = ""
  last_image_topic = ""

  image_sub = None
  img_width = 0
  img_height = 0 

  classes_list = []
  target_detected = False

  pt_namespace = "None"
  pt_connected = False
  has_position_feedback = False
  has_adjustable_speed = False
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
  last_track_time = 0

  class_selected = False
  selected_class = "None"
  last_track_dir = -1
  last_pan_pos = 0
  is_moving = False

  no_object_count = 0
  lost_target_count = 0

  reset_image_topic = False
  app_enabled = False
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


  last_app_enabled = False

  #######################
  ### Node Initialization
  def __init__(self):
    #### APP NODE INIT SETUP ####
    nepi_ros.init_node(name= self.FACTORY_NODE_NAME)
    self.class_name = type(self).__name__
    self.base_namespace = nepi_ros.get_base_namespace()
    self.node_name = nepi_ros.get_node_name()
    self.node_namespace = nepi_ros.get_node_namespace()

    ##############################  
    # Create Msg Class
    self.msg_if = MsgIF(log_name = self.class_name)
    self.msg_if.pub_info("Starting IF Initialization Processes")

    ##############################     
    # Initialize Class Variables

   
    # Message Image to publish when detector not running
    message = "TARGETING NOT ENABLED"
    self.app_ne_img = nepi_img.create_message_image(message)

    message = "WAITING FOR AI DETECTOR TO START"
    self.detector_nr_img = nepi_img.create_message_image(message)

    message = "WAITING FOR TARGET CLASS SELECTION"
     self.no_class_img = nepi_img.create_message_image(message)

    message = "WAITING FOR PANTILT SELECTION"
    self.no_pt_img = nepi_img.create_message_image(message)



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
        'app_enabled': {
            'namespace': self.node_namespace,
            'factory_val': False
        },
        'selected_detector': {
            'namespace': self.node_namespace,
            'factory_val': ''
        },
        'image_fov_vert': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_FOV_VERT_DEG
        },
        'image_fov_horz': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_FOV_HORZ_DEG
        },
        'selected_class': {
            'namespace': self.node_namespace,
            'factory_val': "None"
        },
        'target_q_len': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_Q_LEN
        },
        'target_l_len': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_L_LEN
        },
        'min_ratio': {
            'namespace': self.node_namespace,
            'factory_val':self.FACTORY_MIN_AREA_RATIO
        },
        'pt_namespace': {
            'namespace': self.node_namespace,
            'factory_val': "None"
        },
        'track_update_rate': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TRACK_UPDATE_RATE
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
        'error_goal': {
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
        'errors': {
            'namespace': self.node_namespace,
            'topic': 'errors',
            'msg': TrackingErrors,
            'qsize': 1,
            'latch': True
        },
        'navpose_pub': {
            'namespace': self.node_namespace,
            'topic': 'navpose',
            'msg': NavPoseData,
            'qsize': 1,
            'latch': True
        },
        'tracking_image': {
            'namespace': self.node_namespace,
            'topic': 'tracking_image',
            'msg': Image,
            'qsize': 1,
            'latch': True
        },       
        'status': {
            'namespace': self.node_namespace,
            'topic': 'status',
            'msg': AiPtTrackerStatus,
            'qsize': 1,
            'latch': True
        },
        'PTX_GOHOME_TOPIC': {
            'namespace': self.node_namespace,
            'topic': 'PTX_GOHOME_TOPIC',
            'msg': Empty,
            'qsize': 10,
            'latch': None
        },
        'PTX_SET_SPEED_RATIO_TOPIC': {
            'namespace': self.node_namespace,
            'topic': 'PTX_SET_SPEED_RATIO_TOPIC',
            'msg': Float32,
            'qsize': 10,
            'latch': None
        },
        'PTX_JOG_POSITION_TOPIC': {
            'namespace': self.node_namespace,
            'topic': 'PTX_JOG_POSITION_TOPIC',
            'msg': Image,
            'qsize': 10,
            'latch': None
        }, 
        'PTX_GOTO_PAN_RATIO_TOPIC': {
            'namespace': self.node_namespace,
            'topic': 'PTX_GOTO_PAN_RATIO_TOPIC',
            'msg': Float32,
            'qsize': 10,
            'latch': None
        },
        'PTX_GOTO_TILT_RATIO_TOPIC': {
            'namespace': self.node_namespace,
            'topic': 'PTX_GOTO_TILT_RATIO_TOPIC',
            'msg': Float32,
            'qsize': 10,
            'latch': None
        },
        'PTX_JOG_PAN_TOPIC': {
            'tracking_image': self.node_namespace,
            'topic': 'PTX_JOG_PAN_TOPIC',
            'msg': SingleAxisTimedMove,
            'qsize': 10,
            'latch': None
        }, 
        'PTX_JOG_TILT_TOPIC': {
            'namespace': self.node_namespace,
            'topic': 'PTX_JOG_TILT_TOPIC',
            'msg': SingleAxisTimedMove,
            'qsize': 10,
            'latch': None
        },
        'PTX_SET_SOFT_LIMITS_TOPIC': {
            'namespace': self.node_namespace,
            'topic': 'PTX_SET_SOFT_LIMITS_TOPIC',
            'msg': PanTiltLimits,
            'qsize': 10,
            'latch': None
        },
        'PTX_STOP_TOPIC': {
            'tracking_image': self.node_namespace,
            'topic': 'PTX_STOP_TOPIC',
            'msg': Empty,
            'qsize': 10,
            'latch': None
        }, 
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
            'callback': self.setModelCb, 
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
        'set_track_update_rate': {
            'namespace': self.node_namespace,
            'topic': 'set_track_update_rate',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setTrackUpdateRateCb, 
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
        },                
    }


    # Create Node Class ####################
    self.node_if = NodeClassIF(
                    configs_dict = self.CFGS_DICT,
                    params_dict = self.PARAMS_DICT,
                    pubs_dict = self.PUBS_DICT,
                    subs_dict = self.SUBS_DICT,
                    log_class_name = True
    )

    ready = self.node_if.wait_for_ready()


    # Setup Image IF
    self.image_if = ImageIF(namespace = self.node_namespace, topic = 'tracking_image')

    time.sleep(1)

    ##############################
    self.initCb(do_updates = True)
    # Set up save data and save config services ########################################################
    factory_data_rates= {}
    for d in self.data_products:
        factory_data_rates[d] = [0.0, 0.0, 100.0] # Default to 0Hz save rate, set last save = 0.0, max rate = 100.0Hz
    if 'tracking_image' in self.data_products:
        factory_data_rates['tracking_image'] = [1.0, 0.0, 100.0] 
    self.save_data_if = SaveDataIF(data_product_names = self.data_products, self.node_namespace = factory_data_rates)
    


    ##############################
    # Set up AI Manager Status subscriber
    self.ai_mgr_namespace = os.path.join(self.base_namespace, self.AI_MANAGER_NODE_NAME)
    self.msg_if.pub_info("Waiting for Ai Model Mgr status msg")
    nepi_ros.wait_for_topic(self.ai_mgr_namespace)
    self.nepi_ros.create_subscriber(self.ai_mgr_namespace  + "/status", AiModelMgrStatus, self.aiMgrStatusCb, queue_size = 1)
    while self.ai_mgr_status_msg is None:
      nepi_ros.sleep(1)


    # Start AI Manager Subscribers
    FOUND_OBJECT_TOPIC = self.ai_mgr_namespace  + "/found_object"
    self.nepi_ros.create_subscriber(FOUND_OBJECT_TOPIC, ObjectCount, self.foundObjectCb, queue_size = 1)
    BOUNDING_BOXES_TOPIC = self.ai_mgr_namespace  + "/bounding_boxes"
    self.nepi_ros.create_subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self.objectDetectedCb, queue_size = 1)
    time.sleep(1)    


    ##############################
    ## Start Node Processes

    self.image_if.publish_cv2_image(self.app_ne_img)

    # Set up the timer that start scanning when no objects are detected
    self.msg_if.pub_info("Setting up processes")
    nepi_ros.timer(nepi_ros.ros_duration(self.UPDATER_PROCESS_DELAY), self.updaterCb)
    nepi_ros.timer(nepi_ros.ros_duration(self.SCAN_TRACK_PROCESS_DELAY), self.scanTrackCb)
    nepi_ros.timer(nepi_ros.ros_duration(self.IMG_PUB_PROCESS_DELAY), self.imagePubCb)

    ##############################
    ## Initiation Complete
    self.msg_if.pub_info(" Initialization Complete")
    # Spin forever (until object is detected)
    self.nepi_ros.spin()
    ##############################

  #######################
  ### App Config Functions


  def factoryResetCb(self):
    self.last_image_topic = ""
    self.last_sel_pt = ""

    self.publish_status()


  def initCb(self,do_updates = False):
    self.msg_if.pub_info(" Setting init values to param values")
    if do_updates == True:
      self.resetCb(do_updates)



  def resetCb(self,do_updates = True):
    self.publish_status()


  ###################
  ## Status Publisher
  def publish_status(self):
    status_msg = AiPtTrackerStatus()

    status_msg.app_enabled = self.node_if.get_param('app_enabled')
    status_msg.app_msg = self.app_msg
    
    status_msg.available_detectors_list = sorted(self.detectors_list)
    selected_detector = self.node_if.get_param('selected_detector')
    if selected_detector not in self.detectors_list:
      selected_detector = "None"
    status_msg.selected_detector = selected_detector 
    status_msg.detector_connected = self.detector_connected

    status_msg.image_fov_vert_degs = self.node_if.get_param('image_fov_vert')
    status_msg.image_fov_horz_degs = self.node_if.get_param('image_fov_horz')

    status_msg.available_classes_list = sorted(self.classes_list)
    selected_class = self.node_if.get_param('selected_class')
    if selected_class not in self.classes_list:
      selected_class = "None"
    status_msg.selected_class = selected_class 
    status_msg.target_detected = self.target_detected

    status_msg.pantilt_device = self.pt_namespace
    status_msg.pantilt_connected = self.pt_connected
    status_msg.has_position_feedback = self.has_position_feedback
    status_msg.has_adjustable_speed = self.has_adjustable_speed
    self.pt_status_msg_lock.acquire()
    pt_status_msg = copy.deepcopy(self.pt_status_msg)
    self.pt_status_msg_lock.release()
    if pt_status_msg is not None:
      pan_min = pt_status_msg.yaw_min_softstop_deg
      pan_max = pt_status_msg.yaw_max_softstop_deg
      status_msg.pan_min_max_deg = [pan_min,pan_max]

      tilt_min = pt_status_msg.pitch_min_softstop_deg
      tilt_max = pt_status_msg.pitch_max_softstop_deg
      status_msg.tilt_min_max_deg = [tilt_min,tilt_max]
    else:
      status_msg.pan_min_max_deg = [-180,180]
      status_msg.tilt_min_max_deg = [-180,180]


    min_pan = self.node_if.get_param("min_pan_angle")
    max_pan = self.node_if.get_param("max_pan_angle")
    status_msg.set_pan_min_max_deg = [min_pan,max_pan]

    min_tilt = self.node_if.get_param("min_tilt_angle")
    max_tilt = self.node_if.get_param("max_tilt_angle")
    status_msg.set_tilt_min_max_deg = [min_tilt,max_tilt]
    

    status_msg.track_update_rate_hz = self.node_if.get_param("track_update_rate")
    status_msg.min_area_ratio = self.node_if.get_param("min_area_ratio")
    status_msg.scan_speed_ratio = self.node_if.get_param("scan_speed_ratio")
    status_msg.scan_tilt_offset = self.node_if.get_param("scan_tilt_offset")

    status_msg.track_speed_ratio = self.node_if.get_param("track_speed_ratio")
    status_msg.track_tilt_offset = self.node_if.get_param("track_tilt_offset")

    status_msg.error_goal_min_max_deg = self.MIN_MAX_ERROR_GOAL
    status_msg.error_goal_deg = self.node_if.get_param("error_goal")


    status_msg.target_queue_len = self.node_if.get_param("target_q_len")
    status_msg.target_lost_len = self.node_if.get_param("target_l_len")

    self.target_box_q_lock.acquire()
    box_q = copy.deepcopy(self.target_box_q)      
    self.target_box_q_lock.release()
    status_msg.target_queue_count = len(box_q)

    status_msg.is_scanning = self.is_scanning
    status_msg.is_tracking = self.is_tracking
    status_msg.pan_direction = self.current_scan_dir

    self.node_if.publish_pub('status_pub', status_msg)


        

  def updaterCb(self,timer):
    self.updater()

  def updater(self):
    # Save last image topic for next check
    self.last_image_topic = self.current_image_topic
    ############## DEBUG
    #self.node_if.set_param("app_enabled",True)
    #self.node_if.set_param("pt_namespace","/nepi/s2x/iqr_pan_tilt/ptx")
    #self.node_if.set_param("selected_class","person")
    ############## DEBUG

    update_status = False
    app_enabled = self.node_if.get_param("app_enabled")

    app_msg = ""
    #self.msg_if.pub_warn("Running app update process with app enabled: " + str(app_enabled))
    if app_enabled == False:
      app_msg += "Targeting not enabled"
      '''
      self.current_image_topic = "None"
      if self.image_sub is not None:
        self.msg_if.pub_warn("App Disabled, Unsubscribing from Image topic : " + self.last_image_topic)
        self.image_sub.unregister()
        time.sleep(1)
        self.image_sub = None
      if  self.pt_connected == True:
        self.removePtSubs()
        self.pt_connected = False
      self.last_sel_pt = ""
    '''
    elif self.last_app_enabled != app_enabled:
      update_status = True
    self.last_app_enabled = app_enabled


    # Setup PT subscribers and Publishers if needed
    sel_pt = self.node_if.get_param('pt_namespace')
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
        pt_status_topic=nepi_ros.find_topic(self.pt_status_topic)
        if pt_status_topic == "":
          self.msg_if.pub_warn("PT lost: " + self.pt_status_topic)
          self.pt_connected = False
          self.removePtSubs()
          update_status = True
    if self.pt_connected == True:
      app_msg += ", PanTilt connected"
    else:
      app_msg += ", PanTilt not connected"


    # Update detector info
    ai_mgr_status_response = None
    try:
      ai_mgr_status_response = self.get_ai_mgr_status_service()
      #self.msg_if.pub_info(" Got detector status  " + str(ai_mgr_status_response))
    except Exception as e:
      ai_mgr_status_response = None
      self.msg_if.pub_warn("Failed to call AI MGR STATUS service" + str(e))
      self.detector_running = False
      self.node_if.set_param('selected_detector', "")
      #app_msg += ", AI Detector not connected"
    if ai_mgr_status_response != None:
      #app_msg += ", AI Detector connected"
      status_str = str(ai_mgr_status_response)
      #self.msg_if.pub_warn("got ai manager status: " + status_str)
      self.current_image_topic = ai_mgr_status_response.selected_img_topic
      self.current_detector = ai_mgr_status_response.selected_detector
      self.current_detector_state = ai_mgr_status_response.detector_state
      self.detector_running = self.current_detector_state == "Running"
      if self.detector_running != self.detector_was_running:
        update_status = True
      self.detector_was_running = self.detector_running
      classes_list = ai_mgr_status_response.selected_detector_classes
      if classes_list != self.classes_list:
        self.classes_list = classes_list
        #classes_str = str(self.classes_list)
        #self.msg_if.pub_warn("got ai manager status: " + classes_str)
        update_status = True
      self.classes_list = classes_list
      self.node_if.set_param('selected_detector', self.current_detector)
      #self.msg_if.pub_warn("Got image topics last and current: " + self.last_image_topic + " " + self.current_image_topic)

      # Update Image Topic Subscriber
      if self.detector_running == False:
        app_msg += ", Classifier not running"
        self.target_detected=False
        self.current_image_topic = "None"
      else:
        app_msg += ", Classifier running"

      # Update image topic subs
      #self.msg_if.pub_warn("Current Image Topic: " + self.current_image_topic)
      #self.msg_if.pub_warn("Last Image Topic: " + self.last_image_topic)
      if (self.last_image_topic != self.current_image_topic) or (self.image_sub == None and self.current_image_topic != "None") or self.reset_image_topic == True:
        self.reset_image_topic = False
        image_topic = nepi_ros.find_topic(self.current_image_topic)
        if image_topic == "" and self.current_image_topic != "None":
          self.msg_if.pub_info(" Could not find image update topic: " + self.current_image_topic)
        elif app_enabled == True and image_topic != "":
          self.msg_if.pub_info(" Found detect image update topic : " + image_topic)
          update_status = True
          if self.image_sub != None:
            self.msg_if.pub_info(" Unsubscribing to Image topic : " + self.last_image_topic)
            self.image_sub.unregister()
            time.sleep(1)
            self.image_sub = None
          self.msg_if.pub_info(" Subscribing to Image topic : " + image_topic)
          self.image_sub = self.nepi_ros.create_subscriber(image_topic, Image, self.imageCb, queue_size = 1)
        else:
          self.last_image_topic = ""

        if self.current_image_topic == "None" or self.current_image_topic == "":  # Reset last image topic
          if self.image_sub is not None:
            self.msg_if.pub_warn("Unsubscribing to Image topic : " + self.current_image_topic)
            self.image_sub.unregister()
            time.sleep(1)
            self.image_sub = None
            update_status = True
            time.sleep(1)

    # Check class selection
    class_sel = False
    #self.msg_if.pub_warn("sel class: " + sel_class)
    selected_class = self.node_if.get_param('selected_class')
    if len(self.classes_list) > 0:
      if selected_class  in self.classes_list:
        class_sel = True
      if class_sel == False:
        app_msg += ", Target not selected"
      else:
        app_msg += ", Target selected"
    self.class_selected = class_sel

    # Print a message image if needed
    if app_enabled == False:
      #self.msg_if.pub_warn("Publishing Not Enabled image"
      self.image_if.publish_cv2_image(self.app_ne_img)
    elif self.pt_connected == False:
      self.image_if.publish_cv2_image(self.no_pt_img)
    elif self.detector_running == False:
      self.image_if.publish_cv2_image(self.detector_nr_img)
    elif self.class_selected == False:
      self.image_if.publish_cv2_image(self.no_class_img)

    # Update status app msg
    self.app_msg = app_msg
    if self.app_msg != self.last_app_msg:
      update_status = True
    self.last_app_msg = app_msg
    # Publish status if needed
    if update_status == True:
      #self.msg_if.pub_info(" App update process msg: " + app_msg)
      self.publish_status()
    
  

  def setupPtSubs(self,pt_namespace):
    if pt_namespace is not None:
      if pt_namespace != "None" and pt_namespace != "":
        pt_status_topic = os.path.join(pt_namespace,"/ptx/status")
        #self.msg_if.pub_info("Looking for topic name: " + pt_status_topic)
        self.pt_status_topic=nepi_ros.find_topic(pt_status_topic)
        if self.pt_status_sub is not None:
          self.pt_connected = False
          self.removePtSubs()
        if self.pt_status_topic == "":
          self.pt_namespace = "None"
        if self.pt_status_topic != "":
          self.msg_if.pub_info("Found ptx status topic: " + self.pt_status_topic)
          ptx_namespace = self.pt_status_topic.replace("status","")
          self.msg_if.pub_info("Found ptx namespace: " + ptx_namespace)
          self.pt_namespace = ptx_namespace.split("/ptx")[0]
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
          '''
          ptx_capabilities_service_topic = ptx_namespace + "capabilities_query"
          try:
            ptx_caps_service = nepi_ros.connect_service(ptx_capabilities_service_topic, PTXCapabilitiesQuery)
            time.sleep(1)
            ptx_caps = ptx_caps_service()
            self.has_position_feedback = ptx_caps.absolute_positioning
            self.has_adjustable_speed =  ptx_caps.adjustable_speed
          except Exception as e:
            self.msg_if.pub_warn("Failed to call PTX capabilities service: " + ptx_capabilities_service_topic + " " + str(e))
            self.has_position_feedback = False
            self.has_adjustable_speed =  False
          '''
          ## Create Publishers
          self.send_pt_home_pub = self.nepi_ros.create_publisher(PTX_GOHOME_TOPIC, Empty, queue_size=10)
          self.set_pt_speed_ratio_pub = self.nepi_ros.create_publisher(PTX_SET_SPEED_RATIO_TOPIC, Float32, queue_size=10)
          self.set_pt_position_pub = self.nepi_ros.create_publisher(PTX_JOG_POSITION_TOPIC, PanTiltPosition, queue_size=10)
          self.set_pt_pan_ratio_pub = self.nepi_ros.create_publisher(PTX_GOTO_PAN_RATIO_TOPIC, Float32, queue_size=10)
          self.set_pt_tilt_ratio_pub = self.nepi_ros.create_publisher(PTX_GOTO_TILT_RATIO_TOPIC, Float32, queue_size=10)
          self.set_pt_pan_jog_pub = self.nepi_ros.create_publisher(PTX_JOG_PAN_TOPIC, SingleAxisTimedMove, queue_size=10)
          self.set_pt_tilt_jog_pub = self.nepi_ros.create_publisher(PTX_JOG_TILT_TOPIC, SingleAxisTimedMove, queue_size=10)
          self.set_pt_soft_limits_pub = self.nepi_ros.create_publisher(PTX_SET_SOFT_LIMITS_TOPIC, PanTiltLimits, queue_size=10)
          self.pt_stop_motion_pub = self.nepi_ros.create_publisher(PTX_STOP_TOPIC, Empty, queue_size=10)
          time.sleep(1)
          ## Create Subscribers
          self.msg_if.pub_info("Subscribing to PTX Status Msg: " + self.pt_status_topic)
          self.pt_status_sub = self.nepi_ros.create_subscriber(self.pt_status_topic, PTXStatus, self.ptStatusCb, queue_size = 1)
          #self.pt_connected = True # Set in pt_status callback
      else:
        self.pt_namespace = "None"
        self.pt_connected = False
        self.pan_tilt_errors_deg = [0,0]
    else:
      self.pt_namespace = "None"
      self.pt_connected = False
      self.pan_tilt_errors_deg = [0,0]
      

  def removePtSubs(self):
    if self.pt_status_sub is not None:
      ## Create Class Subscribers
      self.msg_if.pub_info("Unsubscribing to PTX Status Msg: " + self.pt_status_topic)
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

    self.pt_namespace = ""
    self.pt_connected = False
    self.pan_tilt_errors_deg = [0,0]

  #######################
  ### Node Callbacks

  def pubStatusCb(self,msg):
    self.publish_status()

  def appEnableCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    self.node_if.set_param('app_enabled',val)
    self.publish_status()



  def setClassCb(self,msg):
    ##self.msg_if.pub_info(msg)
    selected_class = msg.data
    if selected_class in self.classes_list or selected_class == "None":
      self.node_if.set_param('selected_class',  selected_class)
    self.publish_status()
    self.updater()
    

  def setPtTopicCb(self,msg):
    ##self.msg_if.pub_info(msg)
    pt_topic = msg.data
    if pt_topic == "None":
      pt_topic == ""
    if pt_topic != "":
      pt_topic = nepi_ros.find_topic(pt_topic)
    self.node_if.set_param('pt_namespace',  pt_topic)
    self.publish_status()
    self.updater()


  def setVertFovCb(self,msg):
    ##self.msg_if.pub_info(msg)
    fov = msg.data
    if fov > 0:
      self.node_if.set_param('image_fov_vert',  fov)
    self.publish_status()


  def setHorzFovCb(self,msg):
    ##self.msg_if.pub_info(msg)
    fov = msg.data
    if fov > 0:
      self.node_if.set_param('image_fov_horz',  fov)
    self.publish_status()


  def setTrackUpdateRateCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val < self.MIN_TRACK_UPDATE_RATE:
      val = self.MIN_TRACK_UPDATE_RATE
    if val > self.MAX_TRACK_UPDATE_RATE:
      val = self.MAX_TRACK_UPDATE_RATE
      self.node_if.set_param('track_update_rate',  val)
    self.publish_status()

  def setTargetQLenCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val < 1:
      val = 1
    if val > 20:
      val = 20
    self.node_if.set_param('target_q_len',  val)
    self.publish_status()

  def setTargetLLenCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val < 1:
      val = 1
    if val > 20:
      val = 20
    self.node_if.set_param('target_l_len',  val)
    self.publish_status()


  def setMinAreaCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.node_if.set_param('min_area_ratio',  val)
    self.publish_status()

  def setScanSpeedCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.node_if.set_param('scan_speed_ratio',  val)
    self.publish_status()

  def setScanTiltOffsetCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data

    min_tilt = self.node_if.get_param("min_tilt_angle")
    max_tilt = self.node_if.get_param("max_tilt_angle")
    if val >= min_tilt and val <= max_tilt:
      self.node_if.set_param('scan_tilt_offset',  val)
    self.publish_status()

  def setMinMaxPanCb(self,msg):
    min_pan = msg.start_range
    max_pan = msg.stop_range
    ##self.msg_if.pub_info(msg)
    if min_pan >= -180 and max_pan <= 180 and min_pan < max_pan:
      self.node_if.set_param("min_pan_angle",min_pan)
      self.node_if.set_param("max_pan_angle",max_pan)
    self.publish_status()

  def setMinMaxTiltCb(self,msg):
    min_tilt = msg.start_range
    max_tilt = msg.stop_range
    ##self.msg_if.pub_info(msg)
    if min_tilt >= -180 and max_tilt <= 180 and min_tilt < max_tilt:
      self.node_if.set_param("min_tilt_angle",min_tilt)
      self.node_if.set_param("max_tilt_angle",max_tilt)
    self.publish_status()


  def setTrackSpeedCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.node_if.set_param('track_speed_ratio',  val)
    self.publish_status()

  def setTrackTiltOffsetCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    min_pan = self.node_if.get_param("min_pan_angle")
    max_pan = self.node_if.get_param("max_pan_angle")
    min_tilt = self.node_if.get_param("min_tilt_angle")
    max_tilt = self.node_if.get_param("max_tilt_angle")
    if val >= min_tilt and val <= max_tilt:
      self.node_if.set_param('track_tilt_offset',  val)
    self.publish_status()

  def setErrorGoalCb(self,msg):
    ##self.msg_if.pub_info(msg)
    val = msg.data
    if val < self.MIN_MAX_ERROR_GOAL[0]:
      val = self.MIN_MAX_ERROR_GOAL[0]
    if val > self.MIN_MAX_ERROR_GOAL[1]:
      val = self.MIN_MAX_ERROR_GOAL[1]
    self.node_if.set_param('error_goal',  val)
    self.publish_status()
  #######################
  ### PT Callbacks


  ### Simple callback to get pt pt_status_msg info
  def ptStatusCb(self,pt_status_msg):
    # This is just to get the current pt positions
    if pt_status_msg.yaw_now_deg != self.last_pan_pos:
      self.last_pan_pos = pt_status_msg.yaw_now_deg
      self.is_moving = True
    else:
      self.is_moving = False

    pan_min = pt_status_msg.yaw_min_softstop_deg
    pan_max = pt_status_msg.yaw_max_softstop_deg
    min_pan = self.node_if.get_param("min_pan_angle")
    max_pan = self.node_if.get_param("max_pan_angle")
    if min_pan < pan_min:
      min_pan = pan_min
    if max_pan > pan_max:
      max_pan = pan_max
    self.node_if.set_param("min_pan_angle",min_pan)
    self.node_if.set_param("max_pan_angle",max_pan)
    

    tilt_min = pt_status_msg.pitch_min_softstop_deg
    tilt_max = pt_status_msg.pitch_max_softstop_deg
    min_tilt = self.node_if.get_param("min_tilt_angle")
    max_tilt = self.node_if.get_param("max_tilt_angle")
    if min_tilt < tilt_min:
      min_tilt = tilt_min
    if max_tilt > tilt_max:
      max_tilt = tilt_max
    self.node_if.set_param("min_tilt_angle",min_tilt)
    self.node_if.set_param("max_tilt_angle",max_tilt)

    self.has_position_feedback = pt_status_msg.has_position_feedback
    self.has_adjustable_speed =  pt_status_msg.has_adjustable_speed
    self.cur_speed_ratio = pt_status_msg.speed_ratio
    self.pt_status_msg_lock.acquire()
    self.pt_status_msg = pt_status_msg
    self.pt_status_msg_lock.release()

    self.pt_connected = True
  

  def imagePubCb(self,timer):
    data_product = 'tracking_image'
    app_enabled = self.node_if.get_param("app_enabled")
    if app_enabled and self.image_if is not None and self.detector_running and self.class_selected and self.pt_connected:
      has_subscribers = self.image_if.has_subscribers_check()
      #self.msg_if.pub_warn("Checking for subscribers: " + str(has_subscribers))
      saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
      snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
      should_save = (saving_is_enabled and self.save_data_if.data_product_should_save(data_product)) or snapshot_enabled
      #self.msg_if.pub_warn("Checking for save_: " + str(should_save))

      if has_subscribers or should_save:
        self.img_msg_lock.acquire()
        img_msg = copy.deepcopy(self.img_msg)
        self.img_msg_lock.release()
        if img_msg is not None:
          ros_timestamp = img_msg.header.stamp
          self.target_box_lock.acquire()
          box = copy.deepcopy(self.target_box)      
          self.target_box_lock.release()
          #self.msg_if.pub_warn("Got image to publish")
          if box is not None:
            #self.msg_if.pub_warn("Overlaying target on image: " + str(box))
            current_image_header = img_msg.header
            ros_timestamp = img_msg.header.stamp     
            cv2_img = nepi_img.rosimg_to_cv2img(img_msg).astype(np.uint8)

            #self.msg_if.pub_warn("Box: " + str(box))
            class_name = box.Class

            [xmin,xmax,ymin,ymax] = [box.xmin,box.xmax,box.ymin,box.ymax]
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            class_name = class_name
            class_color = (0,255,0)
            line_thickness = 2
            cv2.rectangle(cv2_img, start_point, end_point, class_color, thickness=line_thickness)

            cv2_shape = cv2_img.shape
            if  cv2_shape[2] == 3:
              encode = 'bgr8'
            else:
              encode = 'mono8'
            self.image_if.publish_cv2_image(cv2_img, timestamp = ros_timestamp)
            # Save Data if Time
            if should_save:
              self.save_data_if.save_img2file(data_product,cv2_img,ros_timestamp,save_check = False)

  ### If object(s) detected, save bounding box info to global
  def objectDetectedCb(self,bounding_boxes_msg):
    app_enabled = self.node_if.get_param("app_enabled")
    selected_class = selected_class = self.node_if.get_param('selected_class')
    target_q_len = self.node_if.get_param("target_q_len")
    target_l_len = self.node_if.get_param("target_l_len")
    min_area_ratio =  self.node_if.get_param("min_area_ratio")
    ros_timestamp = bounding_boxes_msg.header.stamp
    bb_list = bounding_boxes_msg.bounding_boxes
    self.img_height = bounding_boxes_msg.image_height
    self.img_width = bounding_boxes_msg.image_width
    if app_enabled == False:
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

  ### Monitor Output of AI detector to clear detection status
  def foundObjectCb(self,found_obj_msg):
    target_l_len = self.node_if.get_param("target_l_len")
    #Clean Up
    if found_obj_msg.count == 0:      
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


  def scanTrackCb(self,timer):
    self.pt_status_msg_lock.acquire()
    pt_status_msg = copy.deepcopy(self.pt_status_msg)
    self.pt_status_msg_lock.release()    
    was_tracking = copy.deepcopy(self.is_tracking)
    was_scanning = copy.deepcopy(self.is_scanning)
    app_enabled = self.node_if.get_param("app_enabled")
    if app_enabled == False or self.pt_connected == False or pt_status_msg is None:
      #self.msg_if.pub_warn("Scan Track process not ready")
      self.target_detected=False
      self.pan_tilt_errors_deg = [0,0]
      self.is_tracking = False
      self.is_scanning = False
      if was_tracking or was_scanning:
        self.publish_status()
    else:  
      self.target_box_q_lock.acquire()
      box_q = copy.deepcopy(self.target_box_q)      
      self.target_box_q_lock.release()

      min_pan = self.node_if.get_param("min_pan_angle")
      max_pan = self.node_if.get_param("max_pan_angle")
      min_tilt = self.node_if.get_param("min_tilt_angle")
      max_tilt = self.node_if.get_param("max_tilt_angle")

      tilt_cur = pt_status_msg.pitch_now_deg
      tilt_goal = pt_status_msg.pitch_goal_deg

      pan_cur = pt_status_msg.yaw_now_deg
      pan_goal = pt_status_msg.yaw_goal_deg    
        
      #self.msg_if.pub_warn("Length of box queue: " + str(len(box_q)))
      if len(box_q) == 0: ### Scanning if no targets queued
        self.target_detected=False
        self.pan_tilt_errors_deg = [0,0]
        self.is_tracking = False
        self.is_scanning = True
        if was_scanning == False:
          self.start_scanning = True
          self.publish_status()
        scan_speed_ratio = self.node_if.get_param("scan_speed_ratio")
        scan_tilt_offset = self.node_if.get_param("scan_tilt_offset")
        
        # Check tilt limits
        if scan_tilt_offset < min_tilt:
          scan_tilt_offset = min_tilt
        if scan_tilt_offset > max_tilt:
          scan_tilt_offset = max_tilt
        self.node_if.set_param("scan_tilt_offset",scan_tilt_offset)

        if self.has_adjustable_speed == True and self.cur_speed_ratio != scan_speed_ratio:
          try:
            self.node_if.publish_pub('set_pt_speed_ratio_pub', scan_speed_ratio)
          except:
            pass

        if self.has_position_feedback == True:

          if (pan_cur < (min_pan + 5)):
            pan_tilt_pos_msg = PanTiltPosition()
            pan_tilt_pos_msg.yaw_deg = max_pan
            pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
            try:
             self.node_if.publish_pub('set_pt_position_pub', pan_tilt_pos_msg)
             self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
             self.current_scan_dir = 1
             self.publish_status()
            except Exception as E:
              self.msg_if.pub_warn("Driving to max_pan excpetion: " + str(e))
            #self.msg_if.pub_warn("Driving to max_pan")
            self.start_scanning = False

          elif (pan_cur > (max_pan - 5)):
            pan_tilt_pos_msg = PanTiltPosition()
            pan_tilt_pos_msg.yaw_deg = min_pan
            pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
            try:
              self.node_if.publish_pub('set_pt_position_pub', pan_tilt_pos_msg)
              self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
              self.current_scan_dir = -1
              self.publish_status()
            except:
              self.msg_if.pub_warn("Driving to min_pan excpetion: " + str(e))
            #self.msg_if.pub_warn("Driving to min_pan")
            self.start_scanning = False

          elif self.start_scanning == True:
            if self.current_scan_dir > 0:
              pan_tilt_pos_msg = PanTiltPosition()
              pan_tilt_pos_msg.yaw_deg = max_pan
              pan_tilt_pos_msg.pitch_deg = tilt_cur
              try:
                self.node_if.publish_pub('set_pt_position_pub', pan_tilt_pos_msg)
                self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
                self.publish_status()
              except:
                self.msg_if.pub_warn("Starting to max_pan excpetion: " + str(e))
              #self.msg_if.pub_warn("Starting to max_pan")
              self.start_scanning = False
            else:
              pan_tilt_pos_msg = PanTiltPosition()
              pan_tilt_pos_msg.yaw_deg = min_pan
              pan_tilt_pos_msg.pitch_deg = scan_tilt_offset
              try:
                self.node_if.publish_pub('set_pt_position_pub', pan_tilt_pos_msg)
                self.pan_tilt_goal_deg = [pan_tilt_pos_msg.yaw_deg,pan_tilt_pos_msg.pitch_deg]
                self.publish_status()
              except:
                self.msg_if.pub_warn("Starting to min_pan excpetion: " + str(e))
              self.msg_if.pub_warn("Starting to min_pan")
              self.start_scanning = False
        else:
          pass # Add timed jog controls



      else:   # Run Tracking
        self.target_detected=True
        self.is_tracking = True
        self.is_scanning = False
        self.start_scanning = True

        track_delay = float(1)/self.node_if.get_param('track_update_rate')
        ros_time_now = time.time()
        if (ros_time_now - self.last_track_time + self.SCAN_TRACK_PROCESS_DELAY) > track_delay:
          self.last_track_time = ros_time_now 
          error_goal = self.node_if.get_param("error_goal")
          track_speed_ratio = self.node_if.get_param("track_speed_ratio")
          track_tilt_offset = self.node_if.get_param("track_tilt_offset")         
          if self.has_adjustable_speed == True and self.cur_speed_ratio != track_speed_ratio:
            try:
              self.node_if.publish_pub('set_pt_speed_ratio_pub', track_speed_ratio)
            except:
              pass
          #self.msg_if.pub_warn("Error Goal set to: " + str(error_goal))
          #self.msg_if.pub_warn("Got Targets Errors pan tilt: " + str(self.pan_tilt_errors_deg))
          pan_error_q = []
          tilt_error_q = []
          for box in box_q:
            [pan_error,tilt_error] = self.get_target_bearings(box)
            pan_error_q.append(pan_error)
            tilt_error_q.append(tilt_error)
          pan_error_avg = sum(pan_error_q) / len(pan_error_q)
          tilt_error_avg = sum(tilt_error_q) / len(tilt_error_q)
    

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

              #self.msg_if.pub_warn("Current Pos: " + str([pan_cur,tilt_cur]))
              #self.msg_if.pub_warn("Track to Pos: " + str([pan_to_goal,tilt_to_goal]))
              if self.has_position_feedback == True:
                # Send angle goal
                pt_pos_msg = PanTiltPosition()
                pt_pos_msg.yaw_deg = pan_to_goal
                pt_pos_msg.pitch_deg = tilt_to_goal
                if not nepi_ros.is_shutdown():
                  try:
                    self.node_if.publish_pub('set_pt_position_pub', pt_pos_msg)   
                    self.pan_tilt_goal_deg = [pan_to_goal,tilt_to_goal]
                  except:
                    self.msg_if.pub_warn("Tracking to excpetion: " + str(e))
                  #self.msg_if.pub_warn("Tracking to pan tilt: " + str(pan_to_goal) + " " + str(tilt_to_goal))
                else:
                  pass # add timed jog controls

          if was_tracking == False:
            self.publish_status()
    # Publish errors msg
    if pt_status_msg is not None:
      tracking_error_msg = TrackingErrors()
      tracking_error_msg.yaw_cur_deg = pt_status_msg.yaw_now_deg
      tracking_error_msg.yaw_error_deg = self.pan_tilt_errors_deg[0]
      tracking_error_msg.yaw_goal_deg = self.pan_tilt_goal_deg[0]
      tracking_error_msg.pitch_cur_deg = pt_status_msg.pitch_now_deg
      tracking_error_msg.pitch_error_deg = self.pan_tilt_errors_deg[1]
      tracking_error_msg.pitch_goal_deg = self.pan_tilt_goal_deg[1]
      self.node_if.publish_pub('tracking_error_pub', tracking_error_msg)




  def get_target_bearings(self,box):
      target_vert_angle_deg = 0
      target_horz_angle_deg = 0
      if self.img_height != 0 and self.img_width != 0:
        # Iterate over all of the objects and calculate range and bearing data
        image_fov_vert = self.node_if.get_param('image_fov_vert')
        image_fov_horz = self.node_if.get_param('image_fov_horz')
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




  def imageCb(self,img_msg):    
      self.img_msg_lock.acquire()
      self.img_msg = img_msg
      self.img_msg_lock.release()
      

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


