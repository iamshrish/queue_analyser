
import streamlit as st 
import cv2
import csv
import cvzone
from ultralytics import YOLO
import math
from sort import *
import numpy as np
from deepface import DeepFace
import os
import chromadb
import datetime
import time
import re
import threading
import torch
import matplotlib.pyplot as plt
from facelib import AgeGenderEstimator, FaceDetector, EmotionDetector
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding='utf-8')



 
stop_analysis = False
waiting_times_dict = {}

def analyze_age_and_emotion(cropped_head_path, face_detector):
    try:
        img = plt.imread(cropped_head_path)
        # face_detector = FaceDetector()
        faces, boxes, scores, landmarks = face_detector.detect_align(img)
        return faces, True
    except Exception as e:
        print(f"Error analyzing age and emotion: {e}")
        return None, False
    
def analyze_age_and_emotion_deepface(cropped_head_path):
    backends = ['opencv', 'ssd', 'mtcnn', 'fastmtcnn','retinaface', 'mediapipe','yolov8','yunet','centerface',
]
    try:
        objs = DeepFace.analyze(img_path=cropped_head_path, actions=['age', 'gender'], detector_backend = backends[4], enforce_detection = False)
        return objs, True
    except Exception as e:
        print(f"Error analyzing age and emotion: {e}")
        return None, False
    

def write_to_waiting_time_csv(person_id, waiting_time):
    waiting_times_dict = {}

    if os.path.isfile('waiting_times.csv') and os.path.getsize('waiting_times.csv') > 0:
        with open('waiting_times.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                waiting_times_dict[row['ID']] = {
                    'waiting_times': [float(row['Average Engagement Time (min)']) * 60],  
                    'average_waiting_time': float(row['Average Engagement Time (min)']) * 60
                }

    if person_id in waiting_times_dict:
        waiting_times_dict[person_id]['waiting_times'].append(waiting_time)
        average_waiting_time = sum(waiting_times_dict[person_id]['waiting_times']) / len(waiting_times_dict[person_id]['waiting_times'])
        waiting_times_dict[person_id]['average_waiting_time'] = average_waiting_time
    else:
        waiting_times_dict[person_id] = {'waiting_times': [waiting_time], 'average_waiting_time': waiting_time}
    
    estimated_waiting_times = {}
    total_waiting_time = 0
    for id, data in waiting_times_dict.items():
        total_waiting_time += data['average_waiting_time']
        estimated_waiting_times[id] = total_waiting_time
  
  
    with open('waiting_times.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Average Engagement Time (min)', 'Estimated Waiting Time (min)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for person_id, data in waiting_times_dict.items():
            writer.writerow({'ID': person_id, 'Average Engagement Time (min)': round((data['average_waiting_time'] / 60), 2), 'Estimated Waiting Time (min)': round((estimated_waiting_times[person_id] / 60), 2)})


def write_age_emotion_to_csv_webcam(writer, interval_hour, people_in_queue, current_ids, save_folder, face_detector, age_gender_detector):
    id_age_emotion_list = []
    flag = False
    for person_id in current_ids:
        cropped_head_path = os.path.join(save_folder, f"cropped_head_{person_id}.jpg")
        faces, flag = analyze_age_and_emotion(cropped_head_path, face_detector)
        # analysis_result, flag1 = analyze_age_and_emotion_deepface(cropped_head_path)
        
        genders, age1 = age_gender_detector.detect(faces)
        
        age1 = age1[0]
        # if age1 < age2:
        #     age = age1
        # else:
        #     age = age2
        id_age_emotion_list.append({'ID': person_id, 'Age': age1, 'Gender':genders[0]})
        print(id_age_emotion_list)
    
    writer.writerow({
        'Time': interval_hour,
        'People Count': people_in_queue,
        'People Information': '\n'.join([f"ID - {item['ID']}: Age - {item['Age']} years: Gender - {item['Gender']}" for item in id_age_emotion_list])
    })
    print(writer)
    
def write_age_emotion_to_csv(writer, interval_hour, people_in_queue, current_ids, save_folder, face_detector, age_gender_detector):
    id_age_emotion_list = []
    flag = False
    for person_id in current_ids:
        cropped_head_path = os.path.join(save_folder, f"cropped_head_{person_id}.jpg")
        faces, flag = analyze_age_and_emotion(cropped_head_path, face_detector)
        if flag == True:
            genders, age = age_gender_detector.detect(faces)
        age = age[0]
        id_age_emotion_list.append({'ID': person_id, 'Age': age, 'Gender':genders[0]})
    
    writer.writerow({
        'Time': interval_hour,
        'People Count': people_in_queue,
        'People Information': '\n'.join([f"ID - {item['ID']}: Age - {item['Age']} years: Gender - {item['Gender']}" for item in id_age_emotion_list])
    })
    


@st.cache_resource
def load_model():
    return YOLO('model/best.pt').to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))



def load_yolov8_and_process_webcam(cap, confidence, stframe, kpi2_text,kpi3_text, time_interval):
    global stop_analysis
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

   
    number = 0


    em_folder = f"Embeddings{number}"
    save_folder = f"cropped_heads{number}"
    # output_path = f"final_video{number}.avi"
    collection_name = f"person_embeddings{number}"


    os.makedirs(em_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    client = chromadb.PersistentClient(path = em_folder)

    # client.delete_collection(name=collection_name) 
    
    collection = client.get_or_create_collection(
        name = collection_name,
        metadata={"hnsw:space": "cosine"} 
    )

    # cap = cv2.VideoCapture(0)
    # model = YOLO('best.pt').to(device)
    model = load_model()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0

    prev_time = 0
    tracker = Sort(max_age = 40, min_hits = 2)

    person_head_files = {}

    embeddings = {}
    tracker_ids = {}


    people_in_queue = 0
    waiting_times = []

    people_count_intervals = {}


    time_interval = time_interval

    start_time = time.time()
    current_interval_start_time = start_time
    current_interval = start_time
    

    entry_exit_times = {}
    entry_exit = {}

    frame_count = 0
    # frame_skip = 4
    
    
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    age_gender_detector = AgeGenderEstimator()
    
   
    while True:
        if stop_analysis:  

            break
        ret, video = cap.read()
        if not ret:
            break
 
        frame_count += 1
    
        detections = np.empty((0,5))
        time1 = time.time()
        # print("abc", time1)
        result = model.predict(video, device = 0, half = True)
        # print("a", time.time() - time1)
        class_name = ['person']
        for info in result:
            parameters = info.boxes
            for details in parameters:
                x1, y1, x2, y2 = details.xyxy[0]
                conf = details.conf[0]
                conf = math.ceil(conf * 100)
                class_detect = details.cls[0]
                class_detect = int(class_detect)
                class_detect = class_name[class_detect]
                # print("b", time.time() - time1)
                if conf > (confidence * 100):
                    x1 , y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    current_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_detections))


        tracker_result = tracker.update(detections)
        # print("c", time.time() - time1)
        current_ids = []
        for track_result in tracker_result:
            x1, y1, x2, y2, id = track_result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            cropped_head = video[y1-12:y2+12, x1-12:x2+12]

            
            if id is None or id not in tracker_ids or not tracker_ids[id]:
                filename = f"cropped_head_{frame_count}.jpg"
                frame_count += 1
            if id is not None:
                filename = f"cropped_head_{id}.jpg"
            save_path = os.path.join(save_folder, filename)

            if id is not None and id in person_head_files and person_head_files[id] == filename:
                cv2.imwrite(save_path, cropped_head)
            else:
                person_head_files[id] = filename
                cv2.imwrite(save_path, cropped_head)

            cv2.rectangle(video, (x1, y1), (x2, y2), (255, 0 ,255), 3)
            faces, flag = analyze_age_and_emotion(save_path, face_detector)
            emotion, probab = emotion_detector.detect_emotion(faces)
            if emotion == 0:
                cv2.putText(video, last_emotion, (x1+10, y1-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                emotion = emotion[0]
                last_emotion = emotion 

                cv2.putText(video, emotion, (x1+10, y1-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            current_ids.append(id)
            if id not in entry_exit_times:
                    entry_exit_times[id] = {"entry_time": time.time(), "exit_time": None}
            if id in entry_exit_times:
                        entry_exit_times[id]["exit_time"] = time.time()
                        waiting_time = entry_exit_times[id]["exit_time"] - entry_exit_times[id]["entry_time"]
                        waiting_times.append(waiting_time)
                        write_to_waiting_time_csv(id, waiting_time)

        for id in list(entry_exit_times.keys()):
            if id not in current_ids:
                entry_exit_times[id]["exit_time"] = time.time()
                waiting_time = entry_exit_times[id]["exit_time"] - entry_exit_times[id]["entry_time"]
                waiting_times.append(waiting_time)
                write_to_waiting_time_csv(id, waiting_time)
                del entry_exit_times[id]
            


        if len(current_ids) > people_in_queue:
            people_in_queue += 1

        elif len(current_ids) < people_in_queue:
            people_in_queue -= 1

        if waiting_times:
                average_waiting_time = sum(waiting_times) / len(waiting_times)
        else:
            average_waiting_time = 0
        
        average_waiting_time = average_waiting_time / 60 
        current_time = time.time()
        if current_time - current_interval_start_time >= (time_interval/4):
            print("abc")
            current_interval_start_time = current_time
            interval_hour = time.strftime("%H:%M:%S", time.localtime(current_time))
            people_count_intervals[interval_hour] = people_in_queue
            
            with open('people_count_intervals.csv', 'a', newline='') as csvfile:
                fieldnames = ['Time', 'People Count', 'People Information']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                file_size = os.path.getsize('people_count_intervals.csv')
                if file_size == 0:
                    writer.writeheader()

                ids_list = ', '.join([str(i) for i in current_ids])
                writer.writerow({'Time': interval_hour, 'People Count': people_in_queue, 'People Information': ids_list})
                write_age_emotion_to_csv_webcam(writer, interval_hour, people_in_queue, current_ids, save_folder, face_detector, age_gender_detector)
            
        stframe.image(video, channels = 'BGR', use_column_width = True)

        kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{people_in_queue}</h1>", unsafe_allow_html = True)
        kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.2f}'.format(average_waiting_time)}</h1>", unsafe_allow_html = True)
    
    
    cap.release()
    cv2.destroyAllWindows()
    
    
def load_yolov8_and_process_each_frame(video_name, confidence, stframe,  kpi2_text, kpi3_text, time_interval):

    global global_id_counter
    global stop_analysis
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    video_file = video_name

    file_name = os.path.splitext(os.path.basename(video_file))[0]
    # number = re.search(r'\d+', file_name).group()


    em_folder = f"Embeddings_{file_name}"
    save_folder = f"cropped_heads_{file_name}"
    output_path = f"final_video_{file_name}.avi"
    people_count_intervals_file = f"people_count_intervals_{file_name}.txt"
    collection_name = f"person_embeddings_{file_name}"

    
    os.makedirs(em_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)


    client = chromadb.PersistentClient(path = em_folder)


    collection = client.get_or_create_collection(
        name = collection_name,
        metadata={"hnsw:space": "cosine"} 
    )

    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FPS, 1)
    model = load_model()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("fps", fps)




    frame_count = 0

    prev_time = 0
    tracker = Sort(max_age = 45, min_hits = 3)

    person_head_files = {}

    embeddings = {}
    tracker_ids = {}


    people_in_queue = 0
    waiting_times = []

    people_count_intervals = {}


    time_interval = time_interval

    start_time = time.time()
    current_interval_start_time = start_time
    current_interval = start_time


    fps1 = 0
    total_frames = 0
    entry_exit_times = {}
   

    entry_exit = {}
    frame_count = 0
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    age_gender_detector = AgeGenderEstimator()
    while True:
        if stop_analysis:  
            break
        ret, video = cap.read()
        if not ret:
            break
        frame_count += 1

        total_frames += 1
        detections = np.empty((0,5))
        time1 = time.time()

        result = model.predict(video, device = 0, half = True)
       
        class_name = ['person']
        for info in result:
            parameters = info.boxes
            for details in parameters:
                x1, y1, x2, y2 = details.xyxy[0]
                conf = details.conf[0]
                conf = math.ceil(conf * 100)
                class_detect = details.cls[0]
                class_detect = int(class_detect)
                class_detect = class_name[class_detect]
               
                if conf > (confidence * 100):
                    x1 , y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    current_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_detections))


        tracker_result = tracker.update(detections)


        current_ids = []
        for track_result in tracker_result:
            x1, y1, x2, y2, id = track_result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            cropped_head = video[y1-12:y2+12, x1-12:x2+12]

            
            if id is None or id not in tracker_ids or not tracker_ids[id]:
                filename = f"cropped_head_{frame_count}.jpg"
                frame_count += 1
                

            if id is not None:
                filename = f"cropped_head_{id}.jpg"

            save_path = os.path.join(save_folder, filename)
            if id is not None and id in person_head_files and person_head_files[id] == filename:
                cv2.imwrite(save_path, cropped_head)
            else:
                person_head_files[id] = filename
                cv2.imwrite(save_path, cropped_head)

            cv2.rectangle(video, (x1, y1), (x2, y2), (255, 0 ,255), 3)
            # cvzone.putTextRect(video, f'ID - {id}', [x1+8,y1-12], scale = 1, thickness= 2, border = 1)
                
            faces, flag = analyze_age_and_emotion(save_path, face_detector)
            emotion, probab = emotion_detector.detect_emotion(faces)
            if emotion == 0:
                cv2.putText(video, last_emotion, (x1+10, y1-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                emotion = emotion[0]
                last_emotion = emotion 

                cv2.putText(video, emotion, (x1+10, y1-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            current_ids.append(id)
            if id not in entry_exit_times:
                entry_exit_times[id] = {"entry_time": time.time(), "exit_time": None}
            
            if id in entry_exit_times:
                entry_exit_times[id]["exit_time"] = time.time()
                waiting_time = entry_exit_times[id]["exit_time"] - entry_exit_times[id]["entry_time"]
                waiting_times.append(waiting_time)
                write_to_waiting_time_csv(id, waiting_time)

        for id in list(entry_exit_times.keys()):
            if id not in current_ids:
                entry_exit_times[id]["exit_time"] = time.time()
                waiting_time = entry_exit_times[id]["exit_time"] - entry_exit_times[id]["entry_time"]
                waiting_times.append(waiting_time)
                write_to_waiting_time_csv(id, waiting_time)
                del entry_exit_times[id]
    
     
                
        if len(current_ids) > people_in_queue:
            people_in_queue += 1

        elif len(current_ids) < people_in_queue:
            people_in_queue -= 1

        if waiting_times:
            average_waiting_time = sum(waiting_times) / len(waiting_times)
        else:
            average_waiting_time = 0  
        
        average_waiting_time = average_waiting_time / 60    
        current_time = time.time()
        if current_time - current_interval_start_time >= time_interval:
            current_interval_start_time = current_time
            interval_hour = time.strftime("%H:%M:%S", time.localtime(current_time))
            people_count_intervals[interval_hour] = people_in_queue
            # people_count_intervals_file = f"people_count_intervals_{os.path.splitext(os.path.basename(video_name))[0]}.csv"
            
            with open('people_count_intervals.csv', 'a', newline='') as csvfile:
                fieldnames = ['Time', 'People Count', 'People Information']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                file_size = os.path.getsize('people_count_intervals.csv')
                if file_size == 0:
                    writer.writeheader()

                ids_list = ', '.join([str(i) for i in current_ids])
                write_age_emotion_to_csv(writer, interval_hour, people_in_queue, current_ids, save_folder, face_detector,age_gender_detector)
        stframe.image(video, channels='BGR', use_column_width=True)
        
        kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{people_in_queue}</h1>", unsafe_allow_html = True)
        kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.2f}'.format(average_waiting_time)}</h1>", unsafe_allow_html = True)
    

    return people_count_intervals

    cap.release()
    cv2.destroyAllWindows()
