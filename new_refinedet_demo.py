'''
In this example, we will load a RefineDet model and use it to detect objects.
'''

import argparse
import os
import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
# import skimage.io as io

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_label_name(label_map, labels):
    num_labels = len(label_map.item)
    label_names = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == label_map.item[i].label:
                found = True
                label_names.append(label_map.item[i].display_name)
                break
        assert found == True
    return label_names


def show_results(image_file, results, labelmap, threshold=0.6, save_fig=False):
    img = cv2.imread(image_file)

    for i in range(0, results.shape[0]):
        score = round(results[i, 4, 0], 2)

        label = int(results[i, 4, 1])
        name = get_label_name(labelmap, label)[0]

        x0, y0 = np.array(results[i, 0, :]).astype(int)
        x1, y1 = np.array(results[i, 1, :]).astype(int)
        x2, y2 = np.array(results[i, 2, :]).astype(int)
        x3, y3 = np.array(results[i, 3, :]).astype(int)

        coordinates = np.array([[[x0, y0]], [[x1, y1]], [[x2, y2]], [[x3, y3]]])
        x = np.array([x0, x1, x2, x3])
        y = np.array([y0, y1, y2, y3])
        index_x = np.argmax(x)

        img = cv2.drawContours(img, [coordinates], -1, (0, 0, 255), 2)
        cv2.putText(img, str(name) + str(score), (x[index_x]+1, y[index_x]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255))

    print(image_file)
    cv2.imwrite(image_file[:-4]+'_dets.jpg', img)
    print('Saved: ' + image_file[:-4] + '_dets.jpg')

    cv2.imshow(str(image_file), img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_fig', action='store_true')
    args = parser.parse_args()

    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    # load labelmap
    labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # load model
    model_def = 'models/VGGNet/VOC0712/refinedet_vgg16_320x320/deploy.prototxt'
    model_weights = 'models/VGGNet/VOC0712/refinedet_vgg16_320x320 (copy)/VOC0712_refinedet_vgg16_320x320_iter_60000.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    if '320' in model_def:
        img_resize = 320
    else:
        img_resize = 512

    # im_names = os.listdir('examples/images')
    im_names = ['P0001_1200_2000_8400_9200.jpg']

    stride = 20  # 重叠像素
    for im_name in im_names:
        image_file = 'examples/images/' + im_name
        image = caffe.io.load_image(image_file)
        # image = cv2.imread(image_file)

        # 根据图像尺寸和窗口大小获取图片个数
        width, height, channel = image.shape
        w_num = math.ceil((width - stride) / (img_resize - stride))
        h_num = math.ceil((height - stride) / (img_resize - stride))
        img_num = w_num * h_num

        # 晃动窗口读取图片
        result = None
        processed_image = list()
        for w_count in range(w_num):
            for h_count in range(h_num):
                # 临时图片存储
                temp_img = np.zeros([img_resize, img_resize, 3])

                # (img_resize - stride) 窗口大小，stride 重叠像素
                # 窗口在右下角
                if (w_count+1) * (img_resize-stride) > width and (h_count+1) * (img_resize-stride) > height:
                    temp_img[:width-(w_count * (img_resize-stride)), :height-(h_count * (img_resize-stride)), :] =\
                        image[w_count * (img_resize-stride):width,
                              h_count * (img_resize-stride):height,
                              :]

                # 窗口在右边界
                elif (w_count+1) * (img_resize-stride) > width:
                    temp_img[:width-(w_count * (img_resize-stride)), :, :] = \
                        image[w_count * (img_resize-stride):width,
                              h_count * (img_resize-stride):(h_count + 1) * (img_resize-stride) + stride,
                              :]

                # 窗口在上边界
                elif (h_count+1) * (img_resize-stride) > height:
                    temp_img[:, :height - (h_count * (img_resize-stride)), :] = \
                        image[w_count * (img_resize-stride):(w_count + 1) * (img_resize-stride) + stride,
                              h_count * (img_resize-stride):height,
                              :]

                # 窗口在内部
                else:
                    temp_img[:, :, :] = \
                        image[w_count * (img_resize-stride):(w_count + 1) * (img_resize-stride) + stride,
                              h_count * (img_resize-stride):(h_count + 1) * (img_resize-stride) + stride,
                              :]

                elected_image = temp_img * 255
                cv2.imwrite(image_file[:-4] + str(h_count) + str(w_count) + '_erea.jpg', elected_image)

                # 定义transformer参数
                net.blobs['data'].reshape(1, 3, img_resize, img_resize)
                # 定义transformer参数，下方定义尺寸
                transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
                transformer.set_transpose('data', (2, 0, 1))
                transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
                transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
                transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

                # 定义transformer参数预处理

                transformed_image = transformer.preprocess('data', temp_img)

                # 载入数据
                net.blobs['data'].data[...] = transformed_image

                # 调试时debug看这个size，改下面--------------------------------------------
                detections = net.forward()['detection_out']

                # 每幅小图的索引
                result_num = w_count * h_count + h_count

                # 遍历检测框，并转换为斜框
                det_xmin = detections[0, 0, :, 3] * img_resize
                det_ymin = detections[0, 0, :, 4] * img_resize
                det_xmax = detections[0, 0, :, 5] * img_resize
                det_ymax = detections[0, 0, :, 6] * img_resize




                # 根据图像channel first 和 last 调整------------------------------
                # 选取检测框区域
                for box_num in range(detections.shape[2]):
                   if detections[0, 0, box_num, 2] > 0.6:
                        print('find good box')
                        selected_image = temp_img[int(math.floor(det_ymin[box_num])):int(math.ceil(det_ymax[box_num])),
                                                  int(math.floor(det_xmin[box_num])):int(math.ceil(det_xmax[box_num])),
                                                  :] * 255
                        # 进行开运算去噪，并计算最小外界矩形
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        # selected_image = cv2.morphologyEx(selected_image, cv2.MORPH_OPEN, kernel)

                        # selected_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)
                        # print(selected_image)
                        selected_image = cv2.inRange(selected_image, (100, 100, 100), (255, 255, 255))

                        contours, _ = cv2.findContours(selected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours == []:
                            break
                        for i in range(len(contours)):
                            tar_con = contours[i]
                            if contours[i].shape[0] > tar_con.shape[0]:
                                tar_con = contours[i]
                        ro_result = cv2.minAreaRect(tar_con)
                        # 将坐标转换为四点
                        ro_result = cv2.boxPoints(ro_result)
                        ro_result = np.int0(ro_result)


                        # 将坐标映射回原图
                        ro_result[:, 0] = ro_result[:, 0] + int(math.floor(det_xmin[box_num])) + (img_resize - stride) * w_count
                        ro_result[:, 1] = ro_result[:, 1] + int(math.floor(det_ymin[box_num])) + (img_resize - stride) * h_count

                        class_and_conf = [detections[0, 0, box_num, 2], detections[0, 0, box_num, 1]]
                        class_and_conf = np.array(class_and_conf)
                        class_and_conf = np.expand_dims(class_and_conf, axis=0)

                        merge_result = np.concatenate([ro_result, class_and_conf], axis=0)
                        merge_result = np.expand_dims(merge_result, axis=0)

                        result = np.stack(merge_result)


    # show result
        if result is None:
            print('No detection result.')
        else:
            show_results(image_file, result, labelmap, 0.6, save_fig=args.save_fig)






'''
        # 滑动窗口获取的图像数过大，单一批次会OOM，分批次处理
        now_batch = 0
        batch_size = 8
        # 总的检测结果
        total_detections = []
        for now_batch in range(math.ceil(img_num/batch_size)):
            # 获取batch图像
            if (now_batch + 1) * batch_size < img_num:
                batch_image = np.array(processed_image[now_batch * batch_size:
                                                       (now_batch + 1) * batch_size])
                reshape_num = batch_size
            else:
                batch_image =np.array(processed_image[now_batch * batch_size:])
                reshape_num = img_num - now_batch * batch_size
            print(batch_image.shape)
            print(reshape_num)
            # 定义transformer参数
            net.blobs['data'].reshape(reshape_num, 3, img_resize, img_resize)
            # 定义transformer参数预处理
            transformed_image = transformer.preprocess('data', batch_image)
            # 载入数据
            net.blobs['data'].data[...] = transformed_image

            # 调试时debug看这个size，改下面--------------------------------------------
            detections = net.forward()['detection_out']
            total_detections.append(detections)

        # 处理所有检测结果，将其映射回原图
        for w_count in range(w_num):
            for h_count in range(h_num):
                # 每幅小图的索引
                result_num = w_count * h_count + h_count

                # 改这里，把前两个0，0替换成result_num------------------------------
                bbox_num = total_detections.shape(2)

                # 遍历检测框，并转换为斜框
                for selected_region in bbox_num:
                    if total_detections[result_num][0, 0, selected_region, 2] > 0.5:
                        det_xmin = total_detections[result_num][0, 0, selected_region, 3]
                        det_ymin = total_detections[result_num][0, 0, selected_region, 4]
                        det_xmax = total_detections[result_num][0, 0, selected_region, 5]
                        det_ymax = total_detections[result_num][0, 0, selected_region, 6]

                        # 根据图像channel first 和 last 调整------------------------------
                        # 选取检测框区域
                        selected_image = processed_image[result_num, det_xmin:det_xmax, det_ymin:det_ymax, :]

                        # 进行开运算去噪，并计算最小外界矩形
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        selected_image = cv2.morphologyEx(selected_image, cv2.MORPH_OPEN, kernel)
                        selected_image = cv2.inRange(selected_image, [127, 127, 127], [255, 255, 255])
                        ro_result = cv2.minAreaRect(selected_image)
                        # 将坐标转换为四点
                        ro_result = cv2.boxPoints(ro_result)
                        ro_result = np.int0(ro_result)

                        # 将坐标映射回原图
                        ro_result[:, 0] = ro_result[:, 0] + (img_resize - stride) * w_count
                        ro_result[:, 1] = ro_result[:, 1] + (img_resize - stride) * h_count

                        result = np.column_stack([ro_result,
                                                  total_detections[result_num][0, 0, selected_region, 2],
                                                  total_detections[result_num][0, 0, selected_region, 1]])

        # show result
        show_results(image_file, result, labelmap, 0.6, save_fig=args.save_fig)
'''
