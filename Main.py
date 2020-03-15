from find_face import load_face_model, detect_face
from find_face_part import f_p_load_models, detect_face_part
from find_fake import load_fake_models, detect_fake
import os
import time
import pandas



# 모델을 다 불러와서 session 에 올려놓고 시작
model_time = time.time()

# face 모델 불러오기
f_sess, f_detection_graph, f_category_index = load_face_model()
# face part 모델 다 불러오기
f_p_sess, f_p_detection_graph, f_p_category_index = f_p_load_models()
# 가짜 특징 모델 다 불러옴
fake_sess, fake_detection_graph, fake_category_index = load_fake_models()

# 처리할 비디오 경로
folder_path = "G:\\Facebook_Dataset_video\\dfdc_train_part_01\\dfdc_train_01_real\\"
# folder_path = "G:\\Fake_videos\\face_dotnoise_videos\\"
# folder_path = "G:\\face_grid_videos\\"
folder_list = os.listdir(folder_path)

# num = input("영상에서 몇 프레임마다 detect 할지 입력 : ")
# count = input("몇번째 프레임을 확인할지 입력 : ")
num = 1
count = 0
for video_item in folder_list:

    # 영상을 몇 프레임마다 추출할 건지 입력
    number = int(num)
    # 몇번째 프레임을 추출할지 입력
    count_num = int(count)

    # print(video_item)
    PATH_TO_VIDEO = folder_path + video_item
    print(video_item)

    # video에서 얼굴 찾는 모델 불러와서 얼굴 넘기기
    face_list, load_face_model_time, label_face_list = \
        detect_face(PATH_TO_VIDEO, number, count_num,
                    f_sess, f_detection_graph, f_category_index)
    # print("얼굴 찾는 시간 : " + str(round(load_face_model_time)))
    # print(label_face_list)

    # 얼굴에서 눈, 코, 입 찾기
    load_part_model_time, eye_list, nose_list, mouth_list, label_f_p_str = \
        detect_face_part(face_list, PATH_TO_VIDEO, number, count_num,
                         f_p_sess, f_p_detection_graph, f_p_category_index)
    # print("눈코입 찾는 시간 : " + str(round(load_part_model_time)))
    # print(label_f_p_str)



    # 얼굴, 눈, 코, 입 좌표 넘겨서 가짜 특징 찾기
    load_fake_model_time, label_list = \
        detect_fake(face_list, eye_list, nose_list, mouth_list,
                    PATH_TO_VIDEO, number, count_num,
                    fake_sess, fake_detection_graph, fake_category_index,
                    label_face_list, label_f_p_str)

    # label_list = str(video_item)+":"+str(label_percent)+":0"
    # rnn 데이터로 넘기기 위한 str
    # print(str(label_list))
    # df = pandas.DataFrame(str(label_list))
    # df.to_csv('csv_files/'+str(video_item)+'.csv', header=False, index=False)

    # print("총 걸린시간 : " + str(round(load_face_model_time + load_part_model_time)))
