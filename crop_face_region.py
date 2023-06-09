# coding=utf-8
#This code is part of the Addressee Detection paper
# Copyright (c) 2022 Zhejiang Lab
# Written by Fiseha B. <fisehab@zhejianglab.com>

import numpy as np
import pandas as pd
import os
from pathlib import Path
import csv
from utils.data_preparation import *
import argparse
import sys


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="crop face regions")
    #input_Path="../../MUMMER/MuMMER/annotations"
    parser.add_argument(
        "--input_anno", type=str, default="./data/MUMMER_anno/",required=True,
        help="A directory of img  files")
    parser.add_argument(
        "--input_img", type=str, default="./data/images_org_MUMMER/",
        help="out put directory")
    parser.add_argument(
        "--output", type=str, default="./data/croped_face_region/",
        help="out put directory")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    
    assert os.path.isdir(args.input_anno), "Argument '--input_anno' should be a directory name"
    #assert os.path.isdir(args.input_img), "Argument '--input_img' should be a directory name"
    #assert os.path.isdir(args.output), "Argument '--output' should be a directory name"

    
    
    Anno_path=args.input_anno
    image_path=args.input_img
    output_path=args.output
    output_path=args.output   
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    folder = ['kinect','intel']


    for indx in range(10,26):
        im_path=os.path.join(image_path, f"sc{indx}")
        subfolders = [ f.path for f in os.scandir(im_path) if f.is_dir() ]
        for fld in list(subfolders):
            color_path=os.path.join(fld,"color")
           
            for fls in folder:
                camera_path = os.path.join(color_path,fls)
                anno_path_= os.path.join(Anno_path,fls,f"{fld.split('/')[-1]}.xlsx")
                fl=pd.read_excel(anno_path_)
                subjects=fl["subject_id"].unique()
                for subj in subjects:
                    if(subj=='[]'):
                        continue
                    index = fl.index
                    cond = fl["subject_id"]== subj
                    indexs=index[cond]
                    indexs=indexs.tolist()
                    indx_len=len(indexs)
                    temp_indx= []
                    new_indx=[]

                    for indx in range(len(indexs)):
                        time_stamp=round(fl["time_stamps"][indexs[indx]], 2)
                        frame_id=str(fl["frame_id"][indexs[indx]])
                        if str(frame_id).find(".jpg")==-1:
                             frame_id= f"{frame_id}.jpg"

                        full_img_path=os.path.join(camera_path,frame_id)
                        bbox=fl["bbox"][indexs[indx]]
                        out_frame_id=f"{fls}_{fld.split('/')[-1]}:{str(subj)}_{str(time_stamp)}.jpg"
                        save_loc= os.path.join(output_path,out_frame_id)
                        imcrop(full_img_path,bbox,save_loc)
                        print(f"finshed croping {save_loc}")