import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Conv2D, Pool2D
import matplotlib.pyplot as plt
# 对车牌图片进行处理，分割出车牌中的每一个字符并保存

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    #cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img


license_plate = cv_imread('work/车牌.png')
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)
#小于175为0 黑色，大于为255白色
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)  # ret：阈值，binary_plate：根据阈值处理后的图像数据
# 按列统计像素分布 若为0，为间隔
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        result[col] = result[col] + binary_plate[row][col] / 255
# print(result)
# 记录车牌中字符的位置
character_dict = {}
num = 0
i = 0
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        while result[index] != 0:
            index += 1
        character_dict[num] = [i, index - 1]
        num += 1
        i = index
# print(character_dict)
# 将每个字符填充，并存储
characters = []
for i in range(8):
    if i == 2:
        continue
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    # 将单个字符图像填充为170*170
    ndarray = np.pad(binary_plate[:, character_dict[i][0]:character_dict[i][1]], ((0, 0), (int(padding), int(padding))),
                     'constant', constant_values=(0, 0))
    ndarray = cv2.resize(ndarray, (20, 20))
    cv2.imwrite('work/' + str(i) + '.png', ndarray)
    characters.append(ndarray)


