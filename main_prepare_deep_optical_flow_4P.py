import numpy as np
import os
import cv2
import scipy.io as io

def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

def show_flow_hsv(flow, show_style=1):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])#将直角坐标系光流场转成极坐标系

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    #光流可视化的颜色模式
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 #angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    #hsv转bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, hsv

#########################################train################################################
inst = cv2.optflow.createOptFlow_DeepFlow()
# load image and label
batch_size = 75
epoch_num = 50
list_path_real_fake = "./data/DataList/train_RA"
# save path
fea_save_path = "./results/texture_deep_flow_train_data_RA"
if not os.path.exists(fea_save_path):
    os.makedirs(fea_save_path)
# real and fake set
txt_files_real_fake = get_txtlist(list_path_real_fake)
txtCount_real_fake = len(txt_files_real_fake)
for index in range(txtCount_real_fake):

    txt_file_real_fake = txt_files_real_fake[index]
    # 获取list路径与标签
    txt_file_real_fake = txt_file_real_fake.replace("\\", "/")
    txt_path_name = txt_file_real_fake.split('/')[-1]
    save_mat_name = txt_path_name.split('.')[0]
    with open(txt_file_real_fake, 'r') as file:
        txt_lines = file.readlines()
        nSamples = len(txt_lines)

    print('Preparing deep optical flow train data: ', (txtCount_real_fake - 1, index, txt_path_name))
    # read image
    texture_flow_data = []
    for i in range(0, nSamples-1, 5):
        img_path_label = txt_lines[i]
        img_path = img_path_label.split(" ")[0]
        label = float(img_path_label.split(" ")[1])
        if i == 0:
            img_first_rgb_temp = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_first_rgb = cv2.resize(img_first_rgb_temp, (192, 192))
            img_first_hsv = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2HSV)
            img_first_hsv = cv2.resize(img_first_hsv, (192, 192))
            img_first_ycbcr = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2YCrCb)
            img_first_ycbcr = cv2.resize(img_first_ycbcr, (192, 192))
            img_first_gray = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2GRAY)
            img_first_gray = cv2.resize(img_first_gray, (192, 192))
            texture_flow_data_rgb = img_first_rgb
            texture_flow_data_hsv = img_first_hsv
            texture_flow_data_ycbcr = img_first_ycbcr
            texture_flow_data_rgb_hsv_ycbcr = np.dstack((texture_flow_data_rgb, texture_flow_data_hsv, texture_flow_data_ycbcr))
            continue

        img_next = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
        img_next_gray = cv2.resize(img_next_gray, (192, 192))
        # calculate flow
        #inst = cv2.optflow.createOptFlow_DeepFlow()
        flow = inst.calc(img_first_gray, img_next_gray, None)

        [flow_bgr_vis, flow_hsv_vis] = show_flow_hsv(flow, show_style=1)
        #cv2.imshow('BGRimage', flow_bgr_vis)
        #cv2.waitKey(500)

        # stack the first iamge and all optical flow images
        texture_flow_data_rgb = np.dstack((texture_flow_data_rgb, flow_hsv_vis))
        texture_flow_data_hsv = np.dstack((texture_flow_data_hsv, flow_hsv_vis))
        texture_flow_data_ycbcr = np.dstack((texture_flow_data_ycbcr, flow_hsv_vis))
        texture_flow_data_rgb_hsv_ycbcr = np.dstack((texture_flow_data_rgb_hsv_ycbcr, flow_hsv_vis))

    io.savemat((fea_save_path + "/" + save_mat_name + ".mat"), {"texture_flow_data_rgb": texture_flow_data_rgb, "texture_flow_data_hsv": texture_flow_data_hsv,
                                                                "texture_flow_data_ycbcr": texture_flow_data_ycbcr, "texture_flow_data_rgb_hsv_ycbcr": texture_flow_data_rgb_hsv_ycbcr,
                                                                "label": label})

#############################################devel##########################################################
inst = cv2.optflow.createOptFlow_DeepFlow()
# load image and label
batch_size = 75
epoch_num = 50
list_path_real_fake = "./data/DataList/devel_RA"
# save path
fea_save_path = "./results/texture_flow_devel_data_RA"
if not os.path.exists(fea_save_path):
    os.makedirs(fea_save_path)
# real and fake set
txt_files_real_fake = get_txtlist(list_path_real_fake)
txtCount_real_fake = len(txt_files_real_fake)
for index in range(txtCount_real_fake):
    print('Preparing deep optical flow devel data: ', (txtCount_real_fake - 1, index))
    txt_file_real_fake = txt_files_real_fake[index]
    # 获取list路径与标签
    txt_file_real_fake = txt_file_real_fake.replace("\\", "/")
    txt_path_name = txt_file_real_fake.split('/')[-1]
    save_mat_name = txt_path_name.split('.')[0]
    with open(txt_file_real_fake, 'r') as file:
        txt_lines = file.readlines()
        nSamples = len(txt_lines)
    # read image
    texture_flow_data = []
    for i in range(0, nSamples-1, 5):
        img_path_label = txt_lines[i]
        img_path = img_path_label.split(" ")[0]
        label = float(img_path_label.split(" ")[1])
        if i == 0:
            img_first_rgb_temp = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_first_rgb = cv2.resize(img_first_rgb_temp, (192, 192))
            img_first_hsv = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2HSV)
            img_first_hsv = cv2.resize(img_first_hsv, (192, 192))
            img_first_ycbcr = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2YCrCb)
            img_first_ycbcr = cv2.resize(img_first_ycbcr, (192, 192))
            img_first_gray = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2GRAY)
            img_first_gray = cv2.resize(img_first_gray, (192, 192))
            texture_flow_data_rgb = img_first_rgb
            texture_flow_data_hsv = img_first_hsv
            texture_flow_data_ycbcr = img_first_ycbcr
            texture_flow_data_rgb_hsv_ycbcr = np.dstack((texture_flow_data_rgb, texture_flow_data_hsv, texture_flow_data_ycbcr))
            continue

        img_next = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
        img_next_gray = cv2.resize(img_next_gray, (192, 192))
        # calculate flow
        #inst = cv2.optflow.createOptFlow_DeepFlow()
        flow = inst.calc(img_first_gray, img_next_gray, None)

        [flow_bgr_vis, flow_hsv_vis] = show_flow_hsv(flow, show_style=1)
        #cv2.imshow('BGRimage', flow_bgr_vis)
        #cv2.waitKey(500)

        # stack the first iamge and all optical flow images
        texture_flow_data_rgb = np.dstack((texture_flow_data_rgb, flow_hsv_vis))
        texture_flow_data_hsv = np.dstack((texture_flow_data_hsv, flow_hsv_vis))
        texture_flow_data_ycbcr = np.dstack((texture_flow_data_ycbcr, flow_hsv_vis))
        texture_flow_data_rgb_hsv_ycbcr = np.dstack((texture_flow_data_rgb_hsv_ycbcr, flow_hsv_vis))

    io.savemat((fea_save_path + "/" + save_mat_name + ".mat"), {"texture_flow_data_rgb": texture_flow_data_rgb, "texture_flow_data_hsv": texture_flow_data_hsv,
                                                                "texture_flow_data_ycbcr": texture_flow_data_ycbcr, "texture_flow_data_rgb_hsv_ycbcr": texture_flow_data_rgb_hsv_ycbcr,
                                                                "label": label})

#############################################test##########################################################
inst = cv2.optflow.createOptFlow_DeepFlow()
# load image and label
batch_size = 75
epoch_num = 50
list_path_real_fake = "./data/DataList/test_RA"
# save path
fea_save_path = "./results/texture_flow_test_data_RA"
if not os.path.exists(fea_save_path):
    os.makedirs(fea_save_path)
# real and fake set
txt_files_real_fake = get_txtlist(list_path_real_fake)
txtCount_real_fake = len(txt_files_real_fake)
for index in range(txtCount_real_fake):
    print('Preparing deep optical flow test data: ', (txtCount_real_fake - 1, index))
    txt_file_real_fake = txt_files_real_fake[index]
    # 获取list路径与标签
    txt_file_real_fake = txt_file_real_fake.replace("\\", "/")
    txt_path_name = txt_file_real_fake.split('/')[-1]
    save_mat_name = txt_path_name.split('.')[0]
    with open(txt_file_real_fake, 'r') as file:
        txt_lines = file.readlines()
        nSamples = len(txt_lines)
    # read image
    texture_flow_data = []
    for i in range(0, nSamples-1, 5):
        img_path_label = txt_lines[i]
        img_path = img_path_label.split(" ")[0]
        label = float(img_path_label.split(" ")[1])
        if i == 0:
            img_first_rgb_temp = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_first_rgb = cv2.resize(img_first_rgb_temp, (192, 192))
            img_first_hsv = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2HSV)
            img_first_hsv = cv2.resize(img_first_hsv, (192, 192))
            img_first_ycbcr = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2YCrCb)
            img_first_ycbcr = cv2.resize(img_first_ycbcr, (192, 192))
            img_first_gray = cv2.cvtColor(img_first_rgb_temp, cv2.COLOR_BGR2GRAY)
            img_first_gray = cv2.resize(img_first_gray, (192, 192))
            texture_flow_data_rgb = img_first_rgb
            texture_flow_data_hsv = img_first_hsv
            texture_flow_data_ycbcr = img_first_ycbcr
            texture_flow_data_rgb_hsv_ycbcr = np.dstack((texture_flow_data_rgb, texture_flow_data_hsv, texture_flow_data_ycbcr))
            continue

        img_next = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
        img_next_gray = cv2.resize(img_next_gray, (192, 192))
        # calculate flow
        #inst = cv2.optflow.createOptFlow_DeepFlow()
        flow = inst.calc(img_first_gray, img_next_gray, None)

        [flow_bgr_vis, flow_hsv_vis] = show_flow_hsv(flow, show_style=1)
        #cv2.imshow('BGRimage', flow_bgr_vis)
        #cv2.waitKey(500)

        # stack the first iamge and all optical flow images
        texture_flow_data_rgb = np.dstack((texture_flow_data_rgb, flow_hsv_vis))
        texture_flow_data_hsv = np.dstack((texture_flow_data_hsv, flow_hsv_vis))
        texture_flow_data_ycbcr = np.dstack((texture_flow_data_ycbcr, flow_hsv_vis))
        texture_flow_data_rgb_hsv_ycbcr = np.dstack((texture_flow_data_rgb_hsv_ycbcr, flow_hsv_vis))

    io.savemat((fea_save_path + "/" + save_mat_name + ".mat"), {"texture_flow_data_rgb": texture_flow_data_rgb, "texture_flow_data_hsv": texture_flow_data_hsv,
                                                                "texture_flow_data_ycbcr": texture_flow_data_ycbcr, "texture_flow_data_rgb_hsv_ycbcr": texture_flow_data_rgb_hsv_ycbcr,
                                                                "label": label})
