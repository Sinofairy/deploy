import os
import cv2
import numpy as np

def as_num(x):
    y='{:.5f}'.format(x) # 5f表示保留5位小数点的float型
    return(y)

# 非极大值抑制算法（Non-Maximum Suppression，NMS）
# 更详尽的资料请看 https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

def NMS(dets, thresh):

    #x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    in_scores = dets[:, 4]


    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = in_scores.argsort()[::-1]
    # ::-1表示逆序


    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp

# 1. 加载featuremap的输出  针对不同的Head输出 相关参数也不同
# 包含anchor_pred(初始框), 
# bbox_score(框的置信度), 
# bbox_reg(回归框即框的修正量)

data_dir = "infer_output/model_infer_output_58.txt"
num_list = []
with open(data_dir) as f:
    for line in f:
        num_list.extend([int(i) for i in line.split()])
print('length of num: ', len(num_list))

np_data= np.array(num_list).reshape(-1,8)
np_data = np_data * 0.5
#print(np_data)
for i in range(len(np_data)):
    print(i,' : ', np_data[i])
#score = np_data[:,4]
#print(score.shape)
#print(score)
#score.sort()
#print(score)
#score = score[::-1]
#print(score[10])
#for i in range(20):
    #print('score: ', score[i])

#加载box_score
data_dir = "infer_output/model_infer_output_59.txt"
num_list = []
with open(data_dir) as f:
    for line in f:
        num_list.extend([int(i) for i in line.split()])
print('length of score num: ', len(num_list))

score_data= np.array(num_list).reshape(-1,4)
score_data = score_data * 0.000169198
#print(np_data)
score = score_data[:,0]
score = score.reshape(-1,1)
print("len of score: ", len(score))
print('shape of score: ', score.shape)
for i in range(len(score)):
    print(i, ' : ', score[i])

#加载box_reg
data_dir = "infer_output/model_infer_output_60.txt"
num_list = []
with open(data_dir) as f:
    for line in f:
        num_list.extend([int(i) for i in line.split()])
print('length of score num: ', len(num_list))

box_reg = np.array(num_list).reshape(-1,4)
box_reg = box_reg.astype(np.float)
#new_reg = box_reg
for i in range(len(box_reg)):
    # print("before: ", box_reg[i])
    box_reg[i] = box_reg[i] * [3.51308e-5,5.80183e-5,5.06862e-5,8.78977e-5]
    print(i , " :  ", box_reg[i])
#box_reg = box_reg * 0.000169198
#print(np_data)
#score = score_data[:,0]
print("len of box_reg: ", len(box_reg))
for i in range(10):
    print(box_reg[i])

# 加载图片

dets = np_data[:,0:5]
#for i in range(len(np_data)):
    #print(np_data[i])
new_dets = np.empty((0,5),float)
print('shape of dets: ', dets.shape)
box_num = 0
for i in range(len(dets)):
    if  dets[i][0] > 0 and dets[i][1]> 0 and dets[i][2] > 0 and dets[i][3]> 0 and dets[i][4] > 0:
        box_num += 1
        #print(dets[i])
        #print(dets[i].reshape(1,5))
        #print(new_dets.shape)
        if box_num <= 100:
            new_dets = np.append(new_dets,dets[i].reshape(1,5),axis=0)#np.insert(new_dets,len(new_dets),dets[i],axis=0)# =np.delete(dets, i , axis = 0)
        #print(new_dets)
print('shape of processed dets: ', new_dets.shape)
print('length of processed dets: ', len(new_dets))
print('orginal new_dets: ', new_dets)
new_dets[:,0:4] = new_dets[:,0:4] + box_reg[0:len(new_dets),:]
print('after box_reg: ', new_dets)
new_dets[:,4:5] = score[0:len(new_dets),:]
print('after score modified: ', new_dets)
NMS_thresh = 0.6
keep_dets  = NMS(new_dets , NMS_thresh)
print('keep_dets: ', keep_dets)

img = cv2.imread('data/11_1641549436268.jpg') 
box_color = (0,255,0)
#num = 0
for i in range(len(keep_dets)):
    #if box[4] >= score[9]:
    #box = np_data[i]
    #if (box[4] >= score[100]) and (num < 50):
       # num += 1
        #cv2.rectangle(img, (int(box[0]),int( box[1])), (int(box[2]), int(box[3])), color=box_color, thickness=4)
        #print(num)
    box = new_dets[keep_dets[i]]

    print(box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(as_num(box[4]))
    if box[4] >= 0.0:
        cv2.putText(img, text, (int(box[0]),int( box[1])), font, 1, (0,0,255), 1)
        cv2.rectangle(img, (int(box[0]),int( box[1])), (int(box[2]), int(box[3])), color=box_color, thickness=4)
cv2.imwrite('11_1641549436268.jpg', img)