import json
import random


clss_names = ['person', 'taking_photoes', 'phone_call', 'identification','notebook']
video_name=[]
for i in range(100):
    _='video_name_'+str(i)
    video_name.append(_)

Ann={}
for video in video_name:
    _video={}
    for frame_num in range(10):
        fram_name ='xxx_' +str(frame_num)+'.jpg'
        # 当前图片名字_名字要包含frame_number 例如：xxx_5.jpg xxx为10位阿拉伯数字，长度不够前面补0 0000000021_1.jpg
        frame_ann={}
        #当前图片目标数目
        obj_num= random.randint(1,10)
        for obj in range(obj_num):
            _=['left_coordinate','right_coordinate','up_coordinate','down_coordinate']
            _.append(clss_names[random.randint(0,4)])
            frame_ann[obj]=_
        _video[fram_name] = frame_ann
    Ann[video] = _video

with open('标注.json', 'w') as f:
    json.dump(Ann, f)
