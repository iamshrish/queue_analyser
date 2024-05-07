import streamlit as st 
import cv2
# import psutil  
# import GPUtil  
import tempfile
# from yolov8_with_streamlit2 import *
import threading
import pandas as pd
import os
import time
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import json
from streamlit_extras.stylable_container import stylable_container

import warnings
warnings.filterwarnings("ignore")
import base64

# def load_yolov8B():
#     import yolo_with_streamlit2 as yolo1
#     return yolo1

def load_yolov8():
    import yolov8_with_streamlit1 as yolo
    return yolo





monitor_thread = threading.Thread(target=load_yolov8)
monitor_thread.daemon = True
monitor_thread.start()

# thread = threading.Thread(target=load_yolov8B)
# thread.daemon = True
# thread.start()

def stop_analysis_callback():
    global stop_analysis
    stop_analysis = True
    
    
people_count_intervals_data = {}

@st.cache_data
def get_slider_step(unit):
    if unit == 'minutes':
        return 1, 1, 60
    elif unit == 'hours':
        return 1, 1, 24
    
 
    
def main():
    st.set_page_config(
        page_title="Queue Analyser",
        page_icon="🚶‍♂️",
        layout="wide"
    )
    global people_count_intervals_data
    # st.subheader("Queue Analyser")

    st.sidebar.title("Settings")

    st.markdown(
        """
        <style>
        [data-testid = "stSidebar"][aria-expanded="true"] > div:first-child{width: 340px; }
        [data-testid = "stSidebar"][aria-expanded="false"] > div:first-child{width: 350px; margin-left: -340px;}
        </style>
        """,
        unsafe_allow_html=True
    )


    
    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value = 0.50, step = 0.10)
    st.sidebar.markdown('---')
    
    unit = st.sidebar.selectbox("Select Time Interval Unit", ['minutes', 'hours'])
    step, min_value, max_value = get_slider_step(unit)
    time_interval = st.sidebar.slider(f"Enter Time Interval ({unit})", min_value=min_value, max_value=max_value, step=step)
    if unit == 'minutes':
        time_interval = time_interval * 60
    elif unit == 'hours':
        time_interval = time_interval * 3600
        
    st.sidebar.markdown('---')
    use_webcam = st.sidebar.checkbox("Use Webcam")
    # use_url = st.sidebar.checkbox("Use URL")
    # use_webrtc = st.sidebar.checkbox("Use WebRTC")
    # st.sidebar.markdown('---')
    
    st.markdown(
        """
        <style>
        [data-testid = "stSidebar"][aria-expanded="true"] > div:first-child{width: 340px;}
        [data-testid = "stSidebar"][aria-expanded="false"] > div:first-child{width: 350px; margin-left: -340px}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    video_file_buffer = None
    if not use_webcam:
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "asf", "m4v"])
    # print(video_file_buffer.name)
    DEMO_VIDEO = 'test_videos/test3.mp4'
    tfflie = tempfile.NamedTemporaryFile(suffix = '.mp4', delete = False)

 
    st.sidebar.markdown('---')
    if use_webcam:
        cap = cv2.VideoCapture(0)
        st.sidebar.text('Using Webcam')
    # elif use_url:
    #     url = st.sidebar.text_input("Enter URL: ")
    elif video_file_buffer:
        tfflie.write(video_file_buffer.read())
        dem_vid = open(tfflie.name, 'rb')
        demo_bytes = dem_vid.read()
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
    else:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO
        dem_vid = open(tfflie.name, 'rb')
        demo_bytes = dem_vid.read()
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
    st.markdown("<style>.button1 { background-color: green; color: white; }</style>", unsafe_allow_html=True)
    

    st.markdown("<style>.button2 { background-color: red; color: white; }</style>", unsafe_allow_html=True)
    
    # streamlit-extras-0.4.2
    button1, button2 = st.columns([3, 1], gap = "large")
    with button1:
        with stylable_container(
            key="green_button",
            css_styles="""
                button {
                    background-color: green;
                    color: white;
                    border-radius: 20px;
                }
                """,
        ):
            run_analysis_button = st.button("Run Analysis")
    with button2:
        with stylable_container(
            key="red_button",
            css_styles="""
                button {
                    background-color: red;
                    color: white;
                    border-radius: 20px;
                }
                """,
        ):
            stop_analysis_button = st.button("Stop Analysis")
    st.sidebar.markdown('---')
    st.write("#")
    video_column, stats_column = st.columns([3, 1], gap = "large")
    st.write("#")
    with video_column:
        stframe = st.empty()

    with stats_column:
        kpi2, kpi3 = st.columns(2, gap = "large")
        st.markdown("**Queue Length**")
        kpi2_text = st.markdown("0")
        
        st.markdown("**Avg. Time spent (min)**")
        kpi3_text = st.markdown("0")
    # st.write("#")
    def feedback_section():
        if 'likes_count' not in st.session_state:
            st.session_state.likes_count = 0
        if 'dislikes_count' not in st.session_state:
            st.session_state.dislikes_count = 0
        button3, button4 = st.columns(2)
        with button3:
            like_button = st.button(f"Like ({st.session_state.likes_count})", key="like_button")
        with button4:
            dislike_button = st.button(f"Neutral ({st.session_state.dislikes_count})", key="dislike_button")
        
        if like_button:
            st.session_state.likes_count += 1

        if dislike_button:
            st.session_state.dislikes_count += 1
        
    
        
    feedback_section()
    # def start_second_analysis():
    #     yolov8B = load_yolov8B()
    #     time_interval = 60
    #     second_analysis_thread = threading.Thread(target=yolov8B.load_yolov8_and_process_webcam, args=(confidence, time_interval))
    #     second_analysis_thread.start()
          
    # cropped_heads_folder = "cropped_heads0"
    
    # if os.path.exists(cropped_heads_folder) and len(os.listdir(cropped_heads_folder)) >= 1:
    #     time.sleep(15)  
    #     start_second_analysis()
    # else:
    #     threading.Timer(10, start_second_analysis).start()
    # def check_and_start_analysis():
    #     if os.path.exists(cropped_heads_folder) and len(os.listdir(cropped_heads_folder)) >= 1:
    #         start_second_analysis()
    #     else:
    #         threading.Timer(10, check_and_start_analysis).start()

    # check_and_start_analysis()
            
    if run_analysis_button:
        yolov8 = load_yolov8()
        if use_webcam:
            yolov8.load_yolov8_and_process_webcam(cap, confidence, stframe,  kpi2_text, kpi3_text, time_interval)
            st.write("Video processing successful")
        else:
            people_count_intervals_data = yolov8.load_yolov8_and_process_each_frame(tfflie.name, confidence, stframe,  kpi2_text, kpi3_text,time_interval)
            st.write("Video processing successful")
            

    elif stop_analysis_button:
        stop_analysis_callback()
    
    view_res = st.sidebar.button("View Results")
    if view_res:
        st.switch_page("pages/Result.py")   

     
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

