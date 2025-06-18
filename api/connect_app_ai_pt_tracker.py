#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#


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

from nepi_interfaces.msg import SaveDataRate, SaveDataStatus

from nepi_api.node_if import NodeSubscribersIF, NodeClassIF
from nepi_api.messages_if import MsgIF
from nepi_api.system_if import SaveDataIF, StatesIF


from nepi_api.connect_mgr_if_ai_model import ConnectMgrAiModelIF
from nepi_api.data_if import ImageIF

from nepi_api.connect_mgr_if_system import ConnectMgrSystemServicesIF


from nepi_api.messages_if import MsgIF
from nepi_api.connect_node_if import ConnectNodeClassIF
from nepi_api.connect_system_if import ConnectSaveDataIF
from nepi_api.data_if import ImageIF


APP_NODE_NAME = 'app_ai_pt_tracker'

class ConnectAppAiPtTracker:
    msg_if = None
    ready = False
    namespace = '~'

    con_node_if = None

    connected = False
    status_msg = None
    status_connected = False

 
    #######################
    ### IF Initialization
    def __init__(self, 
                namespace = None,
                ):
        ####  IF INIT SETUP ####
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

        if namespace is None:
            namespace = nepi_sdk.create_namespace(self.base_namespace,APP_NODE_NAME)
        else:
            namespace = namespace
        self.namespace = nepi_sdk.get_full_namespace(namespace)


        ##############################   
        ## Node Setup

        # Configs Config Dict ####################
        self.CFGS_DICT = {
                'namespace': self.namespace
        }


        # Services Config Dict ####################
        self.SRVS_DICT = None


        # Publishers Config Dict ####################
        self.PUBS_DICT = {
        'publish_status': {
            'namespace': self.node_namespace,
            'topic': 'publish_status',
            'msg': Empty,
            'qsize': 10,
            'latch': False
        },
        'enable_app': {
            'namespace': self.node_namespace,
            'topic': 'enable_app',
            'msg': Bool,
            'qsize': 1,
            'latch': False
        },
        'set_overlay_labels': {
            'namespace': self.node_namespace,
            'topic': 'set_overlay_labels',
            'msg': Bool,
            'qsize': 10,
            'latch': False
        },
        'set_overlay_clf_name': {
            'namespace': self.node_namespace,
            'topic': 'set_overlay_clf_name',
            'msg': Bool,
            'qsize': 10,
            'latch': False
        },
        'set_max_proc_rate': {
            'namespace': self.node_namespace,
            'topic': 'set_max_proc_rate',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        },
        'set_max_img_rate': {
            'namespace': self.node_namespace,
            'topic': 'set_max_img_rate',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        },
        'set_scan_delay_sec': {
            'namespace': self.node_namespace,
            'topic': 'set_scan_delay_sec',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        },
        'set_overlay_img_name': {
            'namespace': self.node_namespace,
            'topic': 'set_overlay_img_name',
            'msg': Bool,
            'qsize': 10,
            'latch': False
        },
        'set_image_fov_vert': {
            'namespace': self.node_namespace,
            'topic': 'set_image_fov_vert',
            'msg': Float32,
            'qsize': 1,
            'latch': False
        },
        'set_image_fov_horz': {
            'namespace': self.node_namespace,
            'topic': 'set_image_fov_horz',
            'msg': Float32,
            'qsize': 1,
            'latch': False
        },
        'select_detector': {
            'namespace': self.node_namespace,
            'topic': 'select_detector',
            'msg': String,
            'qsize': 1,
            'latch': False
        },
        'select_class': {
            'namespace': self.node_namespace,
            'topic': 'select_class',
            'msg': String,
            'qsize': 1,
            'latch': False
        },
        'set_target_queue_len': {
            'namespace': self.node_namespace,
            'topic': 'set_target_queue_len',
            'msg': Int32,
            'qsize': 10,
            'latch': False
        },
        'set_target_lost_len': {
            'namespace': self.node_namespace,
            'topic': 'set_target_lost_len',
            'msg': Int32,
            'qsize': 10,
            'latch': False
        },
        'set_min_area_ratio': {
            'namespace': self.node_namespace,
            'topic': 'set_min_area_ratio',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        },
        'select_pantilt': {
            'namespace': self.node_namespace,
            'topic': 'select_pantilt',
            'msg': String,
            'qsize': 10,
            'latch': False
        },
        'set_scan_speed_ratio': {
            'namespace': self.node_namespace,
            'topic': 'set_scan_speed_ratio',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        },
        'set_scan_tilt_offset': {
            'namespace': self.node_namespace,
            'topic': 'set_scan_tilt_offset',
            'msg': Float32,
            'latch': False
        },
        'set_min_max_pan_angles': {
            'namespace': self.node_namespace,
            'topic': 'set_min_max_pan_angles',
            'msg': RangeWindow,
            'qsize': 10,
            'latch': False
        },
        'set_min_max_tilt_angles': {
            'namespace': self.node_namespace,
            'topic': 'set_min_max_tilt_angles',
            'msg': RangeWindow,
            'qsize': 10,
            'latch': False
        },
        'set_track_speed_ratio': {
            'namespace': self.node_namespace,
            'topic': 'set_track_speed_ratio',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        },
        'set_track_tilt_offset': {
            'namespace': self.node_namespace,
            'topic': 'set_track_tilt_offset',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        },        
        'set_error_goal_deg': {
            'namespace': self.node_namespace,
            'topic': 'set_error_goal_deg',
            'msg': Float32,
            'qsize': 10,
            'latch': False
        }            
        }


        # Subscribers Config Dict ####################
        self.SUBS_DICT = {
            'status_pub': {
                'namespace': self.node_namespace,
                'topic': 'status',
                'msg': AiPtTrackerStatus,
                'qsize': 1,
                'callback': self._statusCb
            },
            'det_status_pub': {
                'namespace': self.node_namespace,
                'topic': 'detector_status',
                'msg': AiDetectorStatus,
                'qsize': 1,
                'callback': self._statusCb
            },
            'found_object': {
                'msg': ObjectCount,
                'namespace': self.node_namespace,
                'topic': 'found_object',
                'qsize': 1,
                'callback': self._statusCb
            },
            'bounding_boxes': {
                'msg': BoundingBoxes,
                'namespace': self.node_namespace,
                'topic': 'bounding_boxes',
                'qsize': 1,
                'callback': self._statusCb
            },
            'errors': {
                'namespace': self.node_namespace,
                'topic': 'errors',
                'msg': TrackingErrors,
                'qsize': 1,
                'callback': self._statusCb
            }           
        }


        # Create Node Class ####################
        
        self.con_node_if = ConnectNodeClassIF(
                        namespace = self.namespace,
                        configs_dict = self.CFGS_DICT,
                        services_dict = self.SRVS_DICT,
                        pubs_dict = self.PUBS_DICT,
                        subs_dict = self.SUBS_DICT,
                        log_class_name = True,
                        msg_if = self.msg_if
        )

        

        self.con_node_if.wait_for_ready()


        ##############################
        # Complete Initialization
        self.ready = True
        self.msg_if.pub_info("IF Initialization Complete")
        ###############################
    

    #######################
    # Class Public Methods
    #######################


    def get_ready_state(self):
        return self.ready

    def wait_for_ready(self, timout = float('inf') ):
        success = False
        if self.ready is not None:
            self.msg_if.pub_info("Waiting for connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.ready == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.ready == False:
                self.msg_if.pub_info("Failed to Connect")
            else:
                self.msg_if.pub_info("Connected")
        return self.ready  

    def get_namespace(self):
        return self.namespace

    def check_connection(self):
        return self.connected

    def wait_for_connection(self, timout = float('inf') ):
        if self.con_node_if is not None:
            self.msg_if.pub_info("Waiting for connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.connected == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.connected == False:
                self.msg_if.pub_info("Failed to Connect")
            else:
                self.msg_if.pub_info("Connected")
        return self.connected


    def check_status_connection(self):
        return self.status_connected

    def wait_for_status_connection(self, timout = float('inf') ):
        if self.con_node_if is not None:
            self.msg_if.pub_info("Waiting for status connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.status_connected == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.status_connected == False:
                self.msg_if.pub_info("Failed to connect to status msg")
            else:
                self.msg_if.pub_info("Status Connected")
        return self.status_connected

    def get_status_dict(self):
        img_status_dict = None
        if self.status_msg is not None:
            img_status_dict = nepi_sdk.convert_msg2dict(self.status_msg)
        return self.img_status_dict

    def unregister(self):
        self._unsubscribeTopic()

    def publish_status(self):
        pub_name = 'publish_status'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg) 

    def enable_app(self,enable_app):
        pub_name = 'enable_app'
        msg = enable_app
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_overlay_labels(self,overlay_labels):
        pub_name = 'set_overlay_labels'
        msg = overlay_labels
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_track_speed_ratio(self,speed_ratio):
        pub_name = 'set_track_speed_ratio'
        msg = speed_ratio
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_overlay_clf_name(self,clf_name):
        pub_name = 'clf_name'
        msg = set_topic
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_max_proc_rate(self,proc_rate):
        pub_name = 'set_max_proc_rate'
        msg = proc_rate
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_max_img_rate(self,img_rate):
        pub_name = 'set_max_img_rate'
        msg = img_rate
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_scan_delay_sec(self,delay_sec):
        pub_name = 'set_scan_delay_sec'
        msg = delay_sec
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_overlay_img_name(self,img_name):
        pub_name = 'set_overlay_img_name'
        msg = img_name
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_image_fov_vert(self,image_fov):
        pub_name = 'set_image_fov_vert'
        msg = image_fov
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_image_fov_horz(self,image_fov):
        pub_name = 'set_image_fov_horz'
        msg = image_fov
        self.con_node_if.publish_pub(pub_name,msg)  

    def select_detector(self,detector):
        pub_name = 'select_detector'
        msg = detector
        self.con_node_if.publish_pub(pub_name,msg)  

    def select_class(self,select_class):
        pub_name = 'select_class'
        msg = select_class
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_target_queue_len(self,queue_len):
        pub_name = 'set_target_queue_len'
        msg = queue_len
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_target_lost_len(self,lost_len):
        pub_name = 'set_target_lost_len'
        msg = lost_len
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_min_area_ratio(self,area_ratio):
        pub_name = 'set_min_area_ratio'
        msg = area_ratio
        self.con_node_if.publish_pub(pub_name,msg)  

    def select_pantilt(self,pantilt):
        pub_name = 'select_pantilt'
        msg = pantilt
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_scan_speed_ratio(self,speed_ratio):
        pub_name = 'set_scan_speed_ratio'
        msg = speed_ratio
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_scan_tilt_offset(self,tilt_offset):
        pub_name = 'set_scan_tilt_offset'
        msg = tilt_offset
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_min_max_pan_angles(self,pan_angles):
        pub_name = 'set_min_max_pan_angles'
        msg = pan_angles
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_min_max_tilt_angles(self,tilt_angles):
        pub_name = 'set_min_max_tilt_angles'
        msg = tilt_angles
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_track_speed_ratio(self,speed_ratio):
        pub_name = 'set_track_speed_ratio'
        msg = speed_ratio
        self.con_node_if.publish_pub(pub_name,msg)   

    def set_track_tilt_offset(self,tilt_offset):
        pub_name = 'set_track_tilt_offset'
        msg = tilt_offset
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_error_goal_deg(self,goal_deg):
        pub_name = 'set_error_goal_deg'
        msg = goal_deg
        self.con_node_if.publish_pub(pub_name,msg)  

    #################
    ## Save Data Functions

    def get_data_products(self):
        data_products = self.con_save_data_if.get_data_products()
        return data_products

    def get_status_dict(self):
        status_dict = self.con_save_data_if.get_status_dict()
        return status_dict

    def save_data_pub(self,enable):
        self.con_save_data_if.save_data_pub(enable)

    def save_data_prefix_pub(self,prefix):
        self.con_save_data_if.save_data_prefix_pub(prefix)

    def save_data_rate_pub(self,rate_hz, data_product = SaveDataRate.ALL_DATA_PRODUCTS):
        self.con_save_data_if.publish_pub(rate_hz, data_product = SaveDataRate.ALL_DATA_PRODUCTS)

    def snapshot_pub(self):
        self.con_save_data_if.publish_pub()

    def reset_pub(self):
        self.con_save_data_if.publish_pub(pub_name,msg)

    def factory_reset_pub(self):
        pub_name = 'factory_reset'
        msg = Empty()
        self.con_save_data_if.publish_pub(pub_name,msg)

    ###############################
    # Class Private Methods
    ###############################
   

    def _unsubscribeTopic(self):
        success = False
        self.connected = False
        if self.con_node_if is not None:
            self.msg_if.pub_warn("Unregistering topic: " + str(self.namespace))
            try:
                self.con_node_if.unregister_class()
                time.sleep(1)
                self.con_node_if = None
                self.namespace = None
                self.status_connected = False 
                self.data_dict = None
                success = True
            except Exception as e:
                self.msg_if.pub_warn("Failed to unregister image:  " + str(e))
        return success


    def _statusCb(self,status_msg):      
        self.status_connected = True
        self.status_msg = status_msg
