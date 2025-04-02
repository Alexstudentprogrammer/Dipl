import random

import cv2
import json
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from PIL import Image
import PIL.ImageOps as imgOps
import matplotlib.lines as mlines


def perform_Transformation(
        image,
        M,
        original_image,
        x1,
        x1_new,
        y1,
        y1_new,
        x2,
        x2_new,
        y2,
        y2_new,
        x3,
        x3_new,
        y3,
        y3_new,
        x4,
        x4_new,
        y4,
        y4_new,
        number,
        point_x,
        point_y,
        new_point_x,
        new_point_y,

):
    rows, cols, ch = image.shape
    # print(M)
    dst = cv2.warpPerspective(image, M, (cols, rows))
    original_image = np.transpose(original_image, (2, 0, 1))
    dst = np.transpose(dst, (2, 0, 1))

    data_sample = np.concatenate((original_image, dst), axis=0)
    ans = [x1_new - x1, y1_new - y1, x2_new - x2, y2_new - y2, x3_new - x3, y3_new - y3, x4_new - x4, y4_new - y4]

    points = [point_x, point_y, new_point_x, new_point_y]
    #
    tensor_ans = torch.Tensor(ans)
    tensor = torch.Tensor(data_sample)
    point_ans = torch.tensor(points)
    #
    torch.save(tensor, 'test_hom/tensor/' + str(number) + '.pt')
    torch.save(tensor_ans, 'test_ans_hom/tensor/' + str(number) + '.pt')
    torch.save(point_ans, 'test_ans_hom/tensor_img/' + str(number) + '.pt')

    # data_sample = data_sample[3:]
    # plt.imshow(np.transpose(data_sample, (1, 2, 0)))
    # plt.show()


# exit(0)


plt.rcParams['axes.facecolor'] = 'black'

plt.axes()
rectangle = plt.Rectangle((20, 20), 20, 20, fc='black', ec="white", linewidth=3)

plt.xlim(0, 60)
plt.ylim(0, 60)
plt.gca().add_patch(rectangle)

l_1 = mlines.Line2D([20, 25], [25, 25], linewidth=3)
l_2 = mlines.Line2D([20, 25], [35, 35], linewidth=3)
l_3 = mlines.Line2D([25, 25], [25, 35], linewidth=3)

l_1_1 = mlines.Line2D([35, 40], [25, 25], color='red', linewidth=3)
l_2_1 = mlines.Line2D([35, 40], [35, 35], color='red', linewidth=3)
l_3_1 = mlines.Line2D([35, 35], [25, 35], color='red', linewidth=3)

plt.gca().add_line(l_1)
plt.gca().add_line(l_2)
plt.gca().add_line(l_3)

plt.gca().add_line(l_1_1)
plt.gca().add_line(l_2_1)
plt.gca().add_line(l_3_1)

plt.savefig("test.jpg")
plt.close('all')

im = Image.open('test.jpg')
im = im.crop((100, 60, 550, 400))
im1 = np.array(im)
original_image = im1

point_x = 160
point_y = 200
for index in range(1000):
    # point 1
    x_1 = 145
    y_1 = 242

    # point 2
    x_2 = 308
    y_2 = 242

    # point 3
    x_3 = 308
    y_3 = 121

    # point 4
    x_4 = 145
    y_4 = 121

    dx_1 = int(random.uniform(-40, 40))
    dy_1 = int(random.uniform(-40, 40))

    dx_2 = int(random.uniform(-40, 40))
    dy_2 = int(random.uniform(-40, 40))

    dx_3 = int(random.uniform(-40, 40))
    dy_3 = int(random.uniform(-40, 40))

    dx_4 = int(random.uniform(-40, 40))
    dy_4 = int(random.uniform(-40, 40))

    x_1_new = x_1 + dx_1
    y_1_new = y_1 + dy_1

    x_2_new = x_2 + dx_2
    y_2_new = y_2 + dy_2

    x_3_new = x_3 + dx_3
    y_3_new = y_3 + dy_3

    x_4_new = x_4 + dx_4
    y_4_new = y_4 + dy_4

    H, status = cv2.findHomography(np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]),
                                   np.array([[x_1_new, y_1_new], [x_2_new, y_2_new], [x_3_new, y_3_new],
                                             [x_4_new, y_4_new]]),
                                   cv2.RANSAC
                                   )
    point = np.array([point_x, point_y, 1])
    new_point_homogen = np.matmul(H, point)

    new_point_x = new_point_homogen[0] / new_point_homogen[2]
    new_point_y = new_point_homogen[1] / new_point_homogen[2]
    perform_Transformation(
        original_image,
        H,
        original_image,
        x_1,
        x_1_new,
        y_1,
        y_1_new,
        x_2,
        x_2_new,
        y_2,
        y_2_new,
        x_3,
        x_3_new,
        y_3,
        y_3_new,
        x_4,
        x_4_new,
        y_4,
        y_4_new,
        index,
        point_x,
        point_y,
        new_point_x,
        new_point_y
    )
