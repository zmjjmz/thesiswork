#!/usr/bin/env python2
from __future__ import division, print_function
import numpy as np
import cPickle as pickle
import cv2
import matplotlib.pyplot as plt
import sys
import glob
from os.path import join, exists, isdir
from os import mkdir, remove

from matplotlib.widgets import Button, RectangleSelector, RadioButtons

def read_data(folder_root):
    """ Read through folder with format indv/images, collecting images and seg info """
    subfolder_list = glob.glob(join(folder_root,'*'))
    indv_dict = {}
    seg_dict = {}
    for subfolder in subfolder_list:
        pictures = [i for i in glob.glob(join(subfolder,'*')) if not isdir(i)]
        indv = subfolder.split('/')[-1]
        indv_dict[indv] = pictures
        seg_dict[indv] = [collect_segs(pic) for pic in pictures]

    return indv_dict, seg_dict

def collect_segs(pic_fn):
    # lots of assumptions
    segfolder = pic_fn.split('.')[0] + '_segs'
    segmentations = []
    all_masks = glob.glob(join(segfolder,'*.mask.pkl'))
    #all_bgd = glob.glob(join(segfolder,'*.bgd.pkl'))
    #all_fgd = glob.glob(join(segfolder,'*.fgd.pkl'))
    # hacky but it should work so long as the filenames are well formed
    keys = ['mask','fgd','bgd']
    for i in range(len(all_masks)):
        segmentations.append({key:join(segfolder,"%d.%s.pkl" % (i,key)) for key in keys})
    return segmentations


def main_init(datafolder):
    """ Read in the image, show it and set buttons up """
    class IMGUI:
        def __init__(self):
            self.fig = plt.figure(figsize=(15,15))
            self.region_marker = cv2.GC_BGD
            self.apply_mask = lambda img, mask: img*(np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD),cv2.GC_BGD,cv2.GC_FGD).astype('uint8')[:,:,np.newaxis])
            # state for IMGIndex callbacks
            self.indv_dict, self.mask_dict = read_data(datafolder)
            # each iteration of grabcut will store a dictionary of 'mask', 'fgdmod', 'bgdmod', each being a filename for the pickle in the fs
            self.keys_list = list(self.indv_dict.keys())
            self.indv_key = 0
            self.cur_indv = self.keys_list[self.indv_key]
            self.cur_indv_ind = 0
            self.imgax = self.fig.add_subplot(121)
            # state for the segmenter callbacks
            self.segax = self.fig.add_subplot(122)
            self.cur_mask_ind = len(self.mask_dict[self.cur_indv][self.cur_indv_ind])-1
            # general state
            self.redraw_all()

        def open_img(self,imname):
            print("Opening %s" % imname)
            return cv2.cvtColor(cv2.imread(imname),cv2.COLOR_BGR2RGB)
        def open_seg(self,seg_type):
            return pickle.load(open(self.mask_dict[self.cur_indv][self.cur_indv_ind][self.cur_mask_ind][seg_type],'rb'))

        def next(self, event):
            if self.cur_indv_ind == (len(self.indv_dict[self.cur_indv])-1):
                self.indv_key += 1
                if self.indv_key == len(self.keys_list):
                    # hit the end of images
                    print("No more individuals")
                    self.indv_key -= 1
                    return
                self.cur_indv = self.keys_list[self.indv_key]
                self.cur_indv_ind = 0
            else:
                self.cur_indv_ind += 1
            self.cur_mask_ind = len(self.mask_dict[self.cur_indv][self.cur_indv_ind])-1
            self.redraw_all()

        def prev(self, event):
            if self.cur_indv_ind == 0:
                self.indv_key -= 1
                if self.indv_key < 0:
                    print("At first image")
                    self.indv_key += 1
                    return
                self.cur_indv = self.keys_list[self.indv_key]
                self.cur_indv_ind = len(self.indv_dict[self.cur_indv])-1
            else:
                self.cur_indv_ind -= 1
            self.cur_mask_ind = len(self.mask_dict[self.cur_indv][self.cur_indv_ind])-1
            self.redraw_all()

        def change_maskmode(self, label):
            label_dict = {'Foreground':cv2.GC_FGD,'Background':cv2.GC_BGD}
            self.region_marker = label_dict[label]
            print("Marking as %s" % label)

        def apply_segmentation(self, coords):
            # actually take the coords and call grabcut
            # we're going to assume that cur_mask_ind points to the previous mask
            if len(self.mask_dict[self.cur_indv][self.cur_indv_ind]) != 0:
                # first we'll load up the previous mask and bgd/fgd mod (if they exist)
                bgd_mod = self.open_seg('bgd')
                fgd_mod = self.open_seg('fgd')
                mask = self.open_seg('mask')
                # so the idea is now we want to take the mask and the region outlined by the coordinates
                # we want to this in the form lower_y:upper_y,lower_x:upper_x
                mask[coords[1]:coords[3],coords[0]:coords[2]] = self.region_marker
                init = cv2.GC_INIT_WITH_MASK
            else:
                # if not, we'll initialize with rect
                bgd_mod = np.zeros((1,65))
                fgd_mod = np.zeros((1,65))
                mask = np.zeros(self.cur_img_shape[:2],np.uint8)
                init = cv2.GC_INIT_WITH_RECT
            iterations = 2 # TODO: Make this a selector
            cv2.grabCut(self.cur_img, mask, coords, bgd_mod, fgd_mod, iterations, init)
            # then store the mask, fgd_mod, bgd_mod
            self.cur_mask_ind += 1
            seg_dict = self.save_segmentation(mask, fgd_mod, bgd_mod)
            # chop off the old list of segmentations
            old_seg_dicts = self.mask_dict[self.cur_indv][self.cur_indv_ind]
            new_seg_dicts = self.mask_dict[self.cur_indv][self.cur_indv_ind][:self.cur_mask_ind]
            # major TODO: clean up segmentations currently in that directory ahead of this one
            new_seg_dicts.append(seg_dict)
            self.mask_dict[self.cur_indv][self.cur_indv_ind] = new_seg_dicts
            self.cleanup_previous_files(old_seg_dicts)

        def cleanup_previous_files(self, old_seg_dicts):
            # look at the segmentation files past the current index and delete them
            for seg_ind, seg in enumerate(old_seg_dicts):
                # assume self.cur_mask_ind corresponds to the most recent segmentation applied, thus we want to delete the ones greater
                if seg_ind > self.cur_mask_ind:
                    keys = ['mask','fgd','bgd']
                    for key in keys:
                        remove(seg[key])

        def undo_segmentation(self, event):
            # actually just move the mask index back 1, no need to recompute
            self.cur_mask_ind -= 1
            if self.cur_mask_ind < -1:
                print("Already at oldest change")
                self.cur_mask_ind = -1
            else:
                self.redraw_all()

        def redo_segmentation(self, event):
            # move the mask index up 1 (if possible)
            self.cur_mask_ind += 1
            if self.cur_mask_ind >= len(self.mask_dict[self.cur_indv][self.cur_indv_ind]):
                print("Already at newest change")
                self.cur_mask_ind -= 1
            else:
                self.redraw_all()

        def recv_segmentation(self, eclick, erelease):
            # the onselect event that gets the final box
            # get bounding box and (maybe?) convert it to pixel space, then pass on to apply_segmentation
            print("%0.2f, %0.2f "  % (eclick.xdata,eclick.ydata))
            print("%0.2f, %0.2f "  % (erelease.xdata,erelease.ydata))
            if (eclick.xdata > erelease.xdata) and (eclick.ydata > erelease.ydata):
                coords = (int(abs(erelease.xdata)), int(abs(erelease.ydata)), int(abs(eclick.xdata)), int(abs(eclick.ydata)))
            else:
                coords = (int(abs(eclick.xdata)), int(abs(eclick.ydata)), int(abs(erelease.xdata)), int(abs(erelease.ydata)))
            print("SEGMENTING PLEASE WAIT")
            self.apply_segmentation(coords)
            self.redraw_all()

        def save_segmentation(self, mask, fgd_mod, bgd_mod):
            # take current indices (individual, which image, which segmentation) from global scope and figure out where to store
            imgfn = self.indv_dict[self.cur_indv][self.cur_indv_ind]
            # make sure you can write to the folder storing the images...
            subdir = imgfn.split('.')[0] + '_segs'
            if not exists(subdir):
                mkdir(subdir)
            makefn = lambda x: join(subdir,"%d.%s.pkl" % (self.cur_mask_ind, x)) # for ease of parsing later
            keys = ['mask','fgd','bgd']
            store_dict = {key:makefn(key) for key in keys}
            pickle.dump(mask,open(store_dict['mask'],'wb'))
            pickle.dump(fgd_mod,open(store_dict['fgd'],'wb'))
            pickle.dump(bgd_mod,open(store_dict['bgd'],'wb'))
            return store_dict

        def redraw_all(self):
            # open the image, draw it on self.imgax
            self.cur_img = self.open_img(self.indv_dict[self.cur_indv][self.cur_indv_ind])
            self.cur_img_shape = self.cur_img.shape
            self.imgax.set_title("Individual %s Image #%d/%d" % (self.cur_indv, self.cur_indv_ind+1, len(self.indv_dict[self.cur_indv])))
            self.imgax.imshow(self.cur_img)
            # we need to open the mask, which is stored as a numpy pickle on the file system
            self.segax.set_title("Mask #%d on image, draw boxes here" % self.cur_mask_ind if self.cur_mask_ind != -1 else "Raw image, draw boxes here to start grabCut")
            if len(self.mask_dict[self.cur_indv][self.cur_indv_ind]) == 0 or (self.cur_mask_ind == -1):
                # if there is no mask, we're going to just display the image on self.segax
                self.segax.imshow(self.cur_img)
            else:
                # otherwise we'll apply the mask to the image and show that instead
                # policy is to open up and set to the latest mask
                print("Loading segmentation %d" % self.cur_mask_ind)
                mask = self.open_seg('mask')
                self.segax.imshow(self.apply_mask(self.cur_img,mask))
            self.fig.canvas.draw()

        def save_figure(self, event):
            fig_save_dir = "/home/zj1992/work/thesis/example_cuts"
            plt.savefig(join(fig_save_dir, "%s-%d" % (self.cur_indv, self.cur_indv_ind)))
            print("Saved view")

        def exit(self, event):
            print("Exiting")
            sys.exit(0)


    cb = IMGUI()
    prevax = cb.fig.add_axes([0.7, 0.05, 0.1, 0.075])
    nextax = cb.fig.add_axes([0.81, 0.05, 0.1, 0.075])
    undoax = cb.fig.add_axes([0.59, 0.05, 0.1, 0.075])
    redoax = cb.fig.add_axes([0.48, 0.05, 0.1, 0.075])
    saveax = cb.fig.add_axes([0.37, 0.05, 0.1, 0.075])
    exitax = cb.fig.add_axes([0.15, 0.05, 0.1, 0.075])

    bgfgax = cb.fig.add_axes([0.26,0.05,0.1,0.075])


    but_next = Button(nextax,'Next')
    but_next.on_clicked(cb.next)
    but_prev = Button(prevax,'Prev')
    but_prev.on_clicked(cb.prev)
    but_undo = Button(undoax,"Undo\nSegmentation")
    but_undo.on_clicked(cb.undo_segmentation)
    but_redo = Button(redoax,"Redo\nSegmentation")
    but_redo.on_clicked(cb.redo_segmentation)
    but_save = Button(saveax,'Save\nView')
    but_save.on_clicked(cb.save_figure)
    but_exit = Button(exitax,'Exit')
    but_exit.on_clicked(cb.exit)

    radio_bgfg = RadioButtons(bgfgax, ['Foreground','Background'], active=1)
    radio_bgfg.on_clicked(cb.change_maskmode)




    rs = RectangleSelector(cb.segax, cb.recv_segmentation, drawtype='box',
            rectprops={'facecolor':'red', 'edgecolor':'black', 'alpha':0.5, 'fill':False})

    cb.fig.show()
    raw_input()



if __name__ == '__main__':
    filename = "/home/zj1992/windows/work2/datasets/humpbacks"
    if not exists(filename):
        print("MOUNT WINDOWS DUMB DUMB")
        raise OSError
    main_init(filename)


