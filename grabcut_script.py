#!/usr/bin/env python2
from __future__ import division, print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from os.path import join, exists

from matplotlib.widgets import Button, RectangleSelector

def read_data(folder_root):
    """ Read through folder with format indv/images """
    subfolder_list = glob.glob(join(folder_root,'*'))
    indv_dict = {}
    for subfolder in subfolder_list:
        pictures = glob.glob(join(subfolder,'*'))
        indv = subfolder.split('/')[-1]
        indv_dict[indv] = pictures

    return indv_dict

def main_init(datafolder):
    """ Read in the image, show it and set buttons up """
    indv_dict = read_data(datafolder)
    keys_list = list(indv_dict.keys())
    indv_key = 0
    cur_indv = keys_list[indv_key]
    cur_indv_ind = 0
    fig = plt.figure(figsize=(15,15))
    imgax = fig.add_subplot(121)
    segax = fig.add_subplot(122)
    open_img = lambda x: cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB)

    def draw_func(self):
        imgax.set_title("Individual %s Image #%d/%d" % (cur_indv, cur_indv_ind+1, len(indv_dict[cur_indv])))
        imgax.imshow(open_img(indv_dict[cur_indv][cur_indv_ind]))
        fig.canvas.draw()

    class IMGIndex:
        def __init__(self):
            draw_func()

        def next(self, event):
            if cur_indv_ind == (len(indv_dict[cur_indv])-1):
                indv_key += 1
                if indv_key == len(keys_list):
                    # hit the end of images
                    print("No more individuals")
                    indv_key -= 1
                    return
                cur_indv = keys_list[indv_key]
                cur_indv_ind = 0
            else:
                cur_indv_ind += 1
            draw_func()

        def prev(self, event):
            if cur_indv_ind == 0:
                indv_key -= 1
                if indv_key < 0:
                    print("At first image")
                    indv_key += 1
                    return
                cur_indv = keys_list[indv_key]
                cur_indv_ind = len(indv_dict[cur_indv])-1
            else:
                cur_indv_ind -= 1
            draw_func()

    class Segmenter:
        def __init__(self):
            pass
        def draw_segmented(self):
            # show the mask applied to the image
            # if there is no mask, just show the image
            pass
        def apply_segmentation(self, coords): # TODO: add number of iterations for grabcut as parameter
            # actually take the coords and call grabcut
            # if it's the first one use init_with_rect
            # then store the mask
            pass
        def undo_segmentation(self):
            # actually just move the mask index back 1, no need to recompute
            pass
        def redo_segmentation(self):
            # move the mask index up 1 (if possible)
            pass
        def save_segmentation(self):
            # write the segmented mask to disk
            pass
        def recv_segmentation(self):
            # the onselect event that gets the final box
            pass


    cb = IMGIndex()
    prevax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    nextax = fig.add_axes([0.81, 0.05, 0.1, 0.075])

    but_next = Button(nextax,'Next')
    but_next.on_clicked(cb.next)
    but_prev = Button(prevax,'Prev')
    but_prev.on_clicked(cb.prev)

    fig.show()
    raw_input()



if __name__ == '__main__':
    filename = "/home/zj1992/windows/work2/datasets/humpbacks"
    if not exists(filename):
        print("MOUNT WINDOWS DUMB DUMB")
        raise OSError
    main_init(filename)


