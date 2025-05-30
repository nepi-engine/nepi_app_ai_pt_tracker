/*
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
 */
import React, { Component } from "react"
import { observer, inject } from "mobx-react"

import Section from "./Section"
import { Columns, Column } from "./Columns"
import Select, { Option } from "./Select"
import { SliderAdjustment } from "./AdjustmentWidgets"
import Button, { ButtonMenu } from "./Button"
import Label from "./Label"
import Input from "./Input"
import Toggle from "react-toggle"
import Styles from "./Styles"
import BooleanIndicator from "./BooleanIndicator"


import ImageViewer from "./Nepi_IF_ImageViewer"
import NepiIFSaveData from "./Nepi_IF_SaveData"

import {createShortUniqueValues, onDropdownSelectedSendStr, onDropdownSelectedSetState, createMenuListFromStrList, onUpdateSetStateValue, onChangeSwitchSendBoolValue} from "./Utilities"

function round(value, decimals = 0) {
  return Number(value).toFixed(decimals)
  //return value && Number(Math.round(value + "e" + decimals) + "e-" + decimals)
}

@inject("ros")
@observer

class AiPtTrackerApp extends Component {
  constructor(props) {
    super(props)

    this.state = {
		
      appName: 'app_ai_pt_tracker',
	    appNamespace: null,
      baseNamespace: null,

      app_enabled: false,
      app_msg: "Loading",

      max_proc_rate_hz: null,
      max_img_rate_hz: null,

      image_name: "tracking_image",


      show_target_settings: false,

      available_detectors_list: [],
      selected_detector: "None",

      available_classes_list: [],
      selected_class: "None",
      target_detected: false,

      show_image_settings: false,
      image_topic: "None",
      image_fov_vert_degs: null,
      image_fov_horz_degs: null,

            
      show_pt_settings: false,     
      selected_pantilt: "None",
      pantilt_connected: false,
      has_position_feedback: false,
      has_adjustable_speed: false,
      has_auto_pan: false,
      has_auto_tilt: false,

      pan_min: -180,
      pan_max: 180,
      tilt_min: -180,
      tilt_max: 180,
      
      auto_pan_enabled: false,
      set_pan_min: -60,
      set_pan_max: 60,

      auto_tilt_enabled: false,
      set_tilt_min: -30,
      set_tilt_max: 30,

      min_area_ratio: null,

      scan_delay_sec: null,
      scan_speed_ratio: null,
      scan_tilt_offset: null,


      track_speed_ratio: null,
      track_tilt_offset: null,


      error_goal_min_max_deg: [1,20],
      error_goal_deg: null,

      target_q_len: null,
      target_l_len: null,
      target_c_len: null,

      is_scanning: false,
      is_tracking: false,
      pan_direction: 1,

      statusListener: null,
      statusErrorListener: null,
      ptListener: null,
      connected: false,
      needs_update: true,


      pitch_cur_deg: null,
      pitch_error_deg: null,
      pitch_goal_deg: null,
      yaw_cur_deg: null,
      yaw_error_deg: null,
      yaw_goal_deg: null

    }

    this.statusListener = this.statusListener.bind(this)
    this.updateStatusListener = this.updateStatusListener.bind(this)
    this.statusErrorListener = this.statusErrorListener.bind(this)
    this.updateStatusErrorListener = this.updateStatusErrorListener.bind(this)
    this.getAppNamespace = this.getAppNamespace.bind(this)
    this.createDetOptions = this.createDetOptions.bind(this)
    this.createPTXOptions = this.createPTXOptions.bind(this)
    this.onEnterSendInputBoxRangeWindowValue = this.onEnterSendInputBoxRangeWindowValue.bind(this)
    this.onClickToggleShowPtSettings = this.onClickToggleShowPtSettings.bind(this)
    this.onClickToggleShowTargetSettings = this.onClickToggleShowTargetSettings.bind(this)
    this.onClickToggleShowImageSettings = this.onClickToggleShowImageSettings.bind(this)
    this.getDisplayImageInfo = this.getDisplayImageInfo.bind(this)
    

  }


  getAppNamespace(){
    const { namespacePrefix, deviceId} = this.props.ros
    var appNamespace = null
    if (namespacePrefix !== null && deviceId !== null){
      appNamespace = "/" + namespacePrefix + "/" + deviceId + "/" + this.state.appName
      if (this.state.connected === false){
        const pub_status_topic = appNamespace + "/publish_status"
        this.props.ros.sendTriggerMsg(pub_status_topic)
      }
    }
    return appNamespace
  }



  // Callback for handling ROS Status messages
  statusListener(message) {

    const tilt_min_max_deg = message.tilt_min_max_deg
    const pan_min_max_deg = message.pan_min_max_deg

    const set_tilt_min_max_deg = message.set_tilt_min_max_deg
    const set_pan_min_max_deg = message.set_pan_min_max_deg

    this.setState({

      app_enabled: message.enabled,
      app_msg: message.msg,

      max_proc_rate_hz: message.max_proc_rate_hz,
      max_img_rate_hz: message.max_img_rate_hz,
           
      image_topic: message.image_topic,
      image_fov_vert_degs: message.image_fov_vert_degs,
      image_fov_horz_degs: message.image_fov_horz_degs,      

      available_detectors_list: message.available_detectors_list,
      selected_detector: message.selected_detector,

      available_classes_list: message.available_classes_list,
      selected_class: message.selected_class,
      target_detected: message.target_detected,

      selected_pantilt: message.selected_pantilt,
      pantilt_connected: message.pantilt_connected,
      has_position_feedback: message.has_position_feedback,
      has_adjustable_speed: message.has_adjustable_speed,
      has_auto_pan: message.has_auto_pan,
      has_auto_tilt: message.has_auto_tilt,


      pan_min: pan_min_max_deg[0],
      pan_max: pan_min_max_deg[1],
      tilt_min: tilt_min_max_deg[0],
      tilt_max: tilt_min_max_deg[1],
      
      auto_pan_enabled: message.auto_pan_enabled,
      set_pan_min: set_pan_min_max_deg[0],
      set_pan_max: set_pan_min_max_deg[1],

      auto_tilt_enabled: message.auto_tilt_enbled,
      set_tilt_min: set_tilt_min_max_deg[0],
      set_tilt_max: set_tilt_min_max_deg[1],

      min_area_ratio: message.min_area_ratio,


      scan_delay_sec: message.scan_delay_sec,
      scan_speed_ratio: message.scan_speed_ratio,
      scan_tilt_offset: message.scan_tilt_offset,

      track_speed_ratio: message.track_speed_ratio,
      track_tilt_offset: message.track_tilt_offset,

      target_q_len: message.target_queue_len,
      target_l_len: message.target_lost_len,
      target_c_len: message.target_queue_count,
            
      error_goal_min_max_deg: message.error_goal_min_max_deg,
      error_goal_deg: message.error_goal_deg,


      is_scanning: message.is_scanning,
      is_tracking: message.is_tracking,
      pan_direction: message.pan_direction
    
  })

  
  this.setState({
      connected: true
  })


  }
  // Callback for handling ROS Status messages
  statusErrorListener(message) {
    this.setState({
      pitch_cur_deg: message.pitch_cur_deg,
      pitch_error_deg: message.pitch_error_deg,
      pitch_goal_deg: message.pitch_goal_deg,
      yaw_cur_deg: message.yaw_cur_deg,
      yaw_error_deg: message.yaw_error_deg,
      yaw_goal_deg: message.yaw_goal_deg
    
  })


}
    // Function for configuring and subscribing to Status
    updateStatusListener() {
      const namespace = this.getAppNamespace()
      const statusNamespace = namespace + '/status'
      if (this.state.statusListener) {
        this.state.statusListener.unsubscribe()
      }
      var statusListener = this.props.ros.setupStatusListener(
            statusNamespace,
            "nepi_app_ai_pt_tracker/AiPtTrackerStatus",
            this.statusListener
          )
      this.setState({ 
        statusListener: statusListener,
      })
    }

    updateStatusErrorListener() {
      const namespace = this.getAppNamespace()
      const statusNamespace = namespace + '/errors'
      if (this.state.statusErrorListener) {
        this.state.statusErrorListener.unsubscribe()
      }
      var statusErrorListener = this.props.ros.setupStatusListener(
            statusNamespace,
            "nepi_app_ai_pt_tracker/TrackingErrors",
            this.statusErrorListener
          )
      this.setState({ 
        statusErrorListener: statusErrorListener,
      })
    }

  // Lifecycle method called when compnent updates.
  // Used to track changes in the topic
  componentDidUpdate(prevProps, prevState, snapshot) {
    const namespace = this.getAppNamespace()
    const namespace_updated = (prevState.appNamespace !== namespace && namespace !== null)
    if (namespace_updated) {
      if (namespace.indexOf('null') === -1){
        this.setState({appNamespace: namespace})
        this.updateStatusListener()
        this.updateStatusErrorListener()
      } 
    }
  }


  // Lifecycle method called just before the component umounts.
  // Used to unsubscribe to Status message
  componentWillUnmount() {
    if (this.state.statusListener) {
      this.state.statusListener.unsubscribe()
    }
    if (this.state.statusErrorListener) {
      this.state.statusErrorListener.unsubscribe()
    }
  }



// Function for creating topic options for Select input
createPTXOptions() {
  const { ptxUnits} = this.props.ros
  const topics = Object.keys(ptxUnits)
  var i
  var items = []
  items.push(<Option value={"None"}>{"None"}</Option>)
  var unique_names = createShortUniqueValues(topics)
  for (i = 0; i < topics.length; i++) {
    items.push(<Option value={topics[i]}>{unique_names[i]}</Option>)
  }
  return items
}



onEnterSendInputBoxRangeWindowValue(event, topicName, entryName, other_val) {
  const {publishRangeWindow} = this.props.ros
  const appNamespace = this.getAppNamespace()
  const namespace = appNamespace + topicName
  var min = -60
  var max = 60
  if(event.key === 'Enter'){
    const value = parseFloat(event.target.value)
    if (!isNaN(value)){
      if (entryName === "min"){
        min = value
        max = other_val
      }
      else if (entryName === "max"){
        min = other_val
        max = value
      }
      publishRangeWindow(namespace,min,max,false)
    }
    document.getElementById(event.target.id).style.color = Styles.vars.colors.black
  }
}


// Function for creating topic options for Select input
createDetOptions() {
  const topics = this.state.available_detectors_list
  var i
  var items = []
  items.push(<Option value={"None"}>{"None"}</Option>)
  var unique_names = createShortUniqueValues(topics)
  for (i = 0; i < topics.length; i++) {
    items.push(<Option value={topics[i]}>{unique_names[i]}</Option>)
  }
  return items
}



renderApp() {
  const {sendTriggerMsg, sendBoolMsg} = this.props.ros
  const NoneOption = <Option>None</Option>
  
  const pantilt_options = this.createPTXOptions()
  const sel_pantilt = this.state.selected_pantilt
  const pantilt_connected = this.state.pantilt_connected


  const det_options = this.createDetOptions()
  const selectedDet = this.state.selected_detector
  const det_selected= selectedDet !== null && selectedDet !== 'None'

  const selectedClass = this.state.selected_class
  const class_sel = selectedClass !== null && selectedClass !== 'None'

  const connected = this.state.connected === true
  const appNamespace = this.getAppNamespace()


  return (
    <Section title={"PT Tracking App"}>

    <Columns>
      <Column>


      <Columns>
        <Column>

            <div hidden={(connected === true)}>

              <pre style={{ height: "40px", overflowY: "auto" ,fontWeight: 'bold' , color: Styles.vars.colors.Green, textAlign: "left" }}>
                  {"Loading"}
                </pre>

              </div>

              <div hidden={(connected === false)}>

                <Label title="Enable Tracking">
                    <Toggle
                    checked={this.state.app_enabled===true}
                    onClick={() => sendBoolMsg(appNamespace + "/enable_app",!this.state.app_enabled)}>
                    </Toggle>
              </Label>

            </div>

        </Column>
        <Column>

        </Column>
      </Columns>


      <Columns>
        <Column>

        <div hidden={(connected !== true )}>

        <Label title={"Select Detector"}>
          <Select
            id="det_select"
            onChange={(event) => onDropdownSelectedSendStr.bind(this)(event, appNamespace + "/select_detector")}
            value={this.state.selected_det}
          >
            {(det_options.length > 1)
              ? det_options
              : NoneOption}
          </Select>
          </Label>



        <Label title={"Select Target Class"}>
          <Select
            id="class_select"
            onChange={(event) => onDropdownSelectedSendStr.bind(this)(event, appNamespace + "/select_class")}
            value={this.state.selected_class}
          >
            {this.state.available_classes_list
              ? createMenuListFromStrList(this.state.available_classes_list, false, [],['None'],[])
              : NoneOption}
          </Select>
          </Label>



              <Label title={"Select Pan-Tilt Device"}>
              <Select
                id="pt_select"
                onChange={(event) => onDropdownSelectedSendStr.bind(this)(event, appNamespace + "/select_pantilt")}
                value={sel_pantilt}
              >
                {(pantilt_options.length > 1)
                  ? pantilt_options
                  : NoneOption}
              </Select>
            </Label>



                  </div>
            
 

          </Column>
        </Columns>






    <div hidden={(connected !== true )}>

      <Columns>
        <Column>


          <Label title={"Detector Selected"}>
            <BooleanIndicator value={det_selected} />
          </Label>


          <Label title={"Target Class Selected"}>
            <BooleanIndicator value={class_sel} />
          </Label>

          <Label title={"Pan-Tilt Connected"}>
            <BooleanIndicator value={pantilt_connected} />
          </Label>


      </Column>
      <Column>

            <Label title={"Scanning"}>
            <BooleanIndicator value={this.state.is_scanning} />
          </Label>

          <Label title={"Tracking"}>
            <BooleanIndicator value={this.state.is_tracking} />
          </Label>

          <Label title={"Target Detected"}>
            <BooleanIndicator value={this.state.target_detected} />
          </Label>


  
      </Column>
    </Columns>
     

    <Columns>
        <Column>
        <Label title={"Pan Deg"}>
            <Input disabled value={round(this.state.yaw_cur_deg,2)} />
          </Label> 
        <Label title={"Pan Error"}>
            <Input disabled value={round(this.state.yaw_error_deg,2)} />
          </Label> 
          <Label title={"Pan Goal"}>
            <Input disabled value={round(this.state.yaw_goal_deg,2)} />
          </Label> 


          <Label title={"Pan Direction"}>
            <Input disabled value={this.state.pan_direction} />
          </Label>


      </Column>
      <Column>

      <Label title={"Tilt Deg"}>
            <Input disabled value={round(this.state.pitch_cur_deg,2)} />
          </Label>

      <Label title={"Tilt Error"}>
            <Input disabled value={round(this.state.pitch_error_deg,2)} />
          </Label>
          <Label title={"Tilt Goal"}>
            <Input disabled value={round(this.state.pitch_goal_deg,2)} />
          </Label>

          <Label title={"Target Count"}>
            <Input disabled value={this.state.target_c_len} />
          </Label>
  
      </Column>
    </Columns>



      <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

      <Columns>
          <Column>

            <ButtonMenu>
              <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_app")}>{"Reset App"}</Button>
            </ButtonMenu>

            </Column>
          <Column>

              <ButtonMenu>
                <Button onClick={() => sendTriggerMsg(appNamespace + "/save_config")}>{"Save Config"}</Button>
          </ButtonMenu>

          </Column>
          <Column>

          <ButtonMenu>
                <Button onClick={() => sendTriggerMsg( appNamespace + "/reset_config")}>{"Reset Config"}</Button>
          </ButtonMenu>


          </Column>
        </Columns>

        </div>

    </Column>
      </Columns>

    </Section>

  
  )
}


onClickToggleShowPtSettings(){
  const currentVal = this.state.show_pt_settings 
  this.setState({show_pt_settings: !currentVal})
  this.render()
}

renderPtSettings() {
  const set_tilt_min = this.state.set_tilt_min ? this.state.set_tilt_min : -180
  const set_tilt_max = this.state.set_tilt_max ? this.state.set_tilt_max : 180
  const NoneOption = <Option>None</Option>
  const appNamespace = this.getAppNamespace()



  return (
    <Section title={"Pan-Tilt Settings"}>

    <Columns>
      <Column>

 

       <Columns>
          <Column>


            <Label title="Show Settings">
                    <Toggle
                      checked={this.state.show_pt_settings===true}
                      onClick={this.onClickToggleShowPtSettings}>
                    </Toggle>
                  </Label>


             
          </Column>
          <Column>
 
          </Column>
          <Column>

          </Column>
          </Columns>

        <div hidden={(this.state.show_pt_settings === false)}>



                <Columns>
                  <Column>

                            <Label title={"Has Position Feedback"}>
                          <BooleanIndicator value={this.state.has_position_feedback} />
                        </Label>


                </Column>
                <Column>

                            <Label title={"Has Adjustable Speed"}>
                          <BooleanIndicator value={this.state.has_adjustable_speed} />
                        </Label>
          
                  </Column>
                </Columns>

                    <SliderAdjustment
                  title={"Track Update Rate (Hz)"}
                  msgType={"std_msgs/float32"}
                  adjustment={this.state.max_proc_rate_hz}
                  topic={appNamespace + "/set_max_proc_rate_hz"}
                  scaled={1.0}
                  min={1}
                  max={5}
                  tooltip={""}
                  unit={""}
              />


                <Columns>
                  <Column>

                  <div hidden={this.state.has_auto_pan === false}>

                  <Label title="Enable Auto Pan">
                    <Toggle
                      checked={this.state.auto_pan===true}
                      onClick={() => this.onChangeSwitchSendBoolValue("/set_auto_pan_enable",!this.state_auto_pan)}>
                    </Toggle>
                  </Label>



                <Label title={"Set Pan Min"}>
                    <Input id="set_pan_min" 
                      value={this.state.set_pan_min} 
                      onChange={(event) => onUpdateSetStateValue.bind(this)(event,"set_pan_min")} 
                      onKeyDown= {(event) => this.onEnterSendInputBoxRangeWindowValue(event,"/set_min_max_pan_angles","min",this.state.set_pan_max)} />
              </Label>
            

                  <Label title={"Set Pan Max"}>
                    <Input id="set_pan_max" 
                     value={this.state.set_pan_max} 
                      onChange={(event) => onUpdateSetStateValue.bind(this)(event,"set_pan_max")} 
                      onKeyDown= {(event) => this.onEnterSendInputBoxRangeWindowValue(event,"/set_min_max_pan_angles","max",this.state.set_pan_min)} />                      
                  </Label>  

                  </div>

                </Column>
                <Column>

                <div hidden={this.state.has_auto_tilt === false}>

                <Label title="Enable Auto Tilt">
                    <Toggle
                      checked={this.state.auto_pan===true}
                      onClick={() => this.onChangeSwitchSendBoolValue("/set_auto_tilt_enable",!this.state_auto_pan)}>
                    </Toggle>
                  </Label>


                  <Label title={"Set Tilt Min"}>
                    <Input id="set_tilt_min" 
                      value={this.state.set_tilt_min} 
                      onChange={(event) => onUpdateSetStateValue.bind(this)(event,"set_tilt_min")} 
                      onKeyDown= {(event) => this.onEnterSendInputBoxRangeWindowValue(event,"/set_min_max_tilt_angles","min",this.state.set_tilt_max)} />
              </Label>
            

                  <Label title={"Set Tilt Max"}>
                    <Input id="set_tilt_max" 
                     value={this.state.set_tilt_max} 
                      onChange={(event) => onUpdateSetStateValue.bind(this)(event,"set_tilt_max")} 
                      onKeyDown= {(event) => this.onEnterSendInputBoxRangeWindowValue(event,"/set_min_max_tilt_angles","max",this.state.set_tilt_min)} />                      
                  </Label>  

                  </div>

                  </Column>
                </Columns>



              <Columns>
                <Column>
                  <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

                        <label style={{fontWeight: 'bold'}} align={"left"} textAlign={"left"}>
                        {"Scan Settings"}
                      </label>



                      <div hidden={this.state.has_adjustable_speed === false}>

                            <SliderAdjustment
                                title={"Scan Speed Ratio"}
                                msgType={"std_msgs/float32"}
                                adjustment={this.state.scan_speed_ratio}
                                topic={appNamespace + "/set_scan_speed_ratio"}
                                scaled={0.01}
                                min={0}
                                max={100}
                                tooltip={""}
                                unit={"%"}
                            />

                        </div>



                        <div hidden={this.state.has_position_feedback === false}>

                            <SliderAdjustment
                                title={"Scan Tilt Offset (Degs)"}
                                msgType={"std_msgs/float32"}
                                adjustment={this.state.scan_tilt_offset}
                                topic={appNamespace + "/set_scan_tilt_offset"}
                                scaled={1.0}
                                min={set_tilt_min}
                                max={set_tilt_max}
                                tooltip={""}
                                unit={""}
                            />

                        </div>

              </Column>
              <Column>


                          <div style={{ borderTop: "1px solid #ffffff", marginTop: Styles.vars.spacing.medium, marginBottom: Styles.vars.spacing.xs }}/>

                          <label style={{fontWeight: 'bold'}} align={"left"} textAlign={"left"}>
                              {"Track Settings"}
                            </label>
{/*
                          <div hidden={this.state.has_adjustable_speed === false}>

                              <SliderAdjustment
                                  title={"Track Speed Ratio"}
                                  msgType={"std_msgs/float32"}
                                  adjustment={this.state.track_speed_ratio}
                                  topic={appNamespace + "/set_track_speed_ratio"}
                                  scaled={0.01}
                                  min={0}
                                  max={100}
                                  tooltip={""}
                                  unit={"%"}
                              />

                          </div>

  */}

                </Column>
              </Columns>


      </div>

    </Column>
      </Columns>

    </Section>

  
  )
}



onClickToggleShowImageSettings(){
  const currentVal = this.state.show_image_settings 
  this.setState({show_image_settings: !currentVal})
  this.render()
}


renderImageSettings() {
  const NoneOption = <Option>None</Option>
  const appNamespace = this.getAppNamespace()

  return (
    <Section title={"Image Settings"}>


          <Columns>
          <Column>


          <SliderAdjustment
              title={"Image Vertical (Degs)"}
              msgType={"std_msgs/float32"}
              adjustment={this.state.image_fov_vert_degs}
              topic={appNamespace + "/set_image_fov_vert"}
              scaled={1.0}
              min={50}
              max={150}
              tooltip={""}
              unit={""}
          />


          </Column>
          <Column>


          <SliderAdjustment
              title={"Image Horizontal (Degs)"}
              msgType={"std_msgs/float32"}
              adjustment={this.state.image_fov_horz_degs}
              topic={appNamespace + "/set_image_fov_horz"}
              scaled={1.0}
              min={50}
              max={150}
              tooltip={""}
              unit={""}
          />

          </Column>
          </Columns>



    </Section>

  
  )
}


onClickToggleShowTargetSettings(){
  const currentVal = this.state.show_target_settings 
  this.setState({show_target_settings: !currentVal})
  this.render()
}

renderTargetSettings() {
  const {sendBoolMsg} = this.props.ros
  const NoneOption = <Option>None</Option>
  const selectedClass = this.state.selected_class
  const appNamespace = this.getAppNamespace()



  return (
    <Section title={"Target Settings"}>

    <Columns>
      <Column>


      <Columns>
        <Column>


              <Label title="Show Settings">
                    <Toggle
                      checked={this.state.show_target_settings===true}
                      onClick={this.onClickToggleShowTargetSettings}>
                    </Toggle>
                  </Label>

        </Column>
        <Column>

        </Column>
      </Columns>




    <div hidden={(this.state.show_target_settings === false)}>

  


    <Columns>
      <Column>


          <SliderAdjustment
              title={"Target Size Filter"}
              msgType={"std_msgs/float32"}
              adjustment={this.state.min_area_ratio}
              topic={appNamespace + "/set_min_area_ratio"}
              scaled={0.01}
              min={0}
              max={100}
              tooltip={""}
              unit={"%"}
          />

          <SliderAdjustment
              title={"Set Error Goal (Degs)"}
              msgType={"std_msgs/float32"}
              adjustment={this.state.error_goal_deg}
              topic={appNamespace + "/set_error_goal_deg"}
              scaled={1.0}
              min={this.state.error_goal_min_max_deg[0]}
              max={this.state.error_goal_min_max_deg[1]}
              tooltip={""}
              unit={""}
          />





          <SliderAdjustment
              title={"Target Queue Length"}
              msgType={"std_msgs/int32"}
              adjustment={this.state.target_q_len}
              topic={appNamespace + "/set_target_queue_len"}
              scaled={1.0}
              min={1}
              max={10}
              tooltip={""}
              unit={""}
          />


          <SliderAdjustment
              title={"Target Lost Length"}
              msgType={"std_msgs/int32"}
              adjustment={this.state.target_l_len}
              topic={appNamespace + "/set_target_lost_len"}
              scaled={1.0}
              min={1}
              max={10}
              tooltip={""}
              unit={""}
          />

      </Column>
    </Columns>

  </div>



    </Column>
      </Columns>

    </Section>

  
  )
}



getDisplayImageInfo(){
  const { namespacePrefix, deviceId} = this.props.ros
  var namespace = "/" + namespacePrefix + "/" + deviceId + "/" + this.state.appName + '/' + this.state.image_name
  var text = "AI PanTilt Tracking"
  return [namespace,text]

}




  render() {
    if (this.state.needs_update === true){
      this.setState({needs_update: false})
    }
    const connected = this.state.connected === true
    const appNamespace = (connected) ? this.getAppNamespace() : null
    const [imageNamespace, imageText] = this.getDisplayImageInfo()

    return (

      <div style={{ display: 'flex' }}>

        <div style={{ width: '55%' }}>


              <ImageViewer
              imageTopic={imageNamespace}
              title={imageText}
              hideQualitySelector={false}
            />

            <div hidden={!connected}>

            <NepiIFSaveData
              saveNamespace={appNamespace}
              title={"Nepi_IF_SaveData"}
            />

              </div>


        </div>


        <div style={{ width: '5%' }}>
          {}
        </div>


        <div style={{ width: '40%' }}>

              {this.renderApp()}

              <div hidden={(this.state.connected === false)} >
              {this.renderImageSettings()}

              {this.renderPtSettings()}                

                {this.renderTargetSettings()}
              </div>

        </div>
      </div>



      )
    }  



}

export default AiPtTrackerApp
