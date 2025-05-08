# AI PanTilt Tracker Status Message

float32 max_proc_rate_hz
float32 max_img_rate_hz

string[] available_detectors
string selected_detector

float32 image_fov_vert_degs 
float32 image_fov_horz_degs 

string selected_pantilt
bool pantilt_connected
bool has_position_feedback
bool has_adjustable_speed
float32[] tilt_min_max_deg
float32[] pan_min_max_deg
float32[] set_tilt_min_max_deg
float32[] set_pan_min_max_deg

float32 scan_delay_sec
float32 scan_speed_ratio
float32 scan_tilt_offset

float32 track_speed_ratio
float32 track_tilt_offset

float32[] error_goal_min_max_deg
float32 error_goal_deg

float32 min_area_ratio

int32 target_queue_len
int32 target_lost_len
int32 target_queue_count

bool is_scanning
bool is_tracking
int32 pan_direction

string msg
bool target_detected

