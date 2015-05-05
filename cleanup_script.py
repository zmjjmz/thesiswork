#!/usr/bin/env python2
# script to remove all segmentations, be careful
import glob
from os.path import join, isdir
from os import remove
from shutil import rmtree
segfolder = '/home/zj1992/windows/work2/datasets/humpbacks'
viewsfolder = '/home/zj1992/work/thesis/example_cuts'

all_views = glob.glob(join(viewsfolder,'*'))
for i in all_views:
    remove(i)
all_indv = glob.glob(join(segfolder,'*'))
for indv in all_indv:
    all_segs = glob.glob(join(indv,'*_segs'))
    for seg in all_segs:
        if isdir(seg):
            #print("Removing: %s (y/n)" % seg)
            #confirm = raw_input()
            #if confirm == "y":
            rmtree(seg)
            print("Removed %s" % seg)


