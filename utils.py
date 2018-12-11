# -*- coding:utf-8 -*-

import os
import numpy as np
from matplotlib import image
    
def read_anime_avatar():
    """Read avatar dataset into a ndarray format.
    """
    data_path = r'/data0/xingyun/anime_avatar/avatar'
    # data_path = r'F:\DATASETS\anime_avatar\avatar'
    img_names = os.listdir(data_path)

    imgs = []
    cnt = 1
    for img_name in img_names:
        curr_img_path = data_path + '/' + img_name
        print("%5d: %s is reading." % (cnt, curr_img_path))
        img = image.imread(curr_img_path)
        img = img/127.5 - 1
        imgs.append(img)
        cnt += 1
        if cnt % 5000 == 0:
            file_name = '/data0/xingyun/anime_avatar/per_5000/avatar_data_'+ str(cnt)+'.npy'
            np.save(file_name, imgs)
            imgs = []
    # file_name = '/data0/xingyun/anime_avatar/avatar_data.npy'
    # np.save(file_name, imgs)

if __name__ == '__main__':
    read_anime_avatar()