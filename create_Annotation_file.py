# coding=utf-8
#This code is part of the Addressee Detection paper
# Copyright (c) 2022 Zhejiang Lab
# Written by Fiseha B. <fisehab@zhejianglab.com>

import numpy as np
import os
import sys
import time
import argparse
import pandas as pd
from utils.data_preparation import *



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create single annotation file for single scenario")
    #input_Path="../../MUMMER/MuMMER/annotations"
    parser.add_argument(
        "--input", type=str, required=True,
        help="A directory of annotation  files")
    parser.add_argument(
        "--output", type=str, default="./data/original_anno/",
        help="out put directory")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
        
    assert os.path.isdir(args.input), "Argument '--input' should be a directory name"
    assert os.path.isdir(args.input), "Argument '--output' should be a directory name"

    anno_folders = ['intel','kinect']
    
    out_put=args.output   
    if os.path.exists(out_put):
        os.makedirs(out_put, exist_ok=True)
    
    for i in range(10,26):
        sc_path_name= f"sc{str(i)}"
        input_path=os.path.join(args.input,sc_path_name)
   
        for item in anno_folders:
              for dir_ in  os.listdir(input_path):
                full_path=os.path.join(input_path,dir_,"color", item)
                annoation_names =  sorted(os.listdir(full_path)) 
            
                index_anno=0
                anno_id=[]
                bboxes=[]
                subject_ids=[]
                time_stamps=[]
                flag=True
                cnt_sec=0
                t_stamp=0
                total_index_anno=[]
                
                for filename in annoation_names:
                    anno_name = os.path.join(full_path,filename)
                    [bbox, sub_ids] = read_annoation_file(anno_name)
                    index_anno+=1
                    cnt_sec+=1
                    t_stamp=cnt_sec/15.0
                    if len(bbox) ==0:
                        anno_id.append(filename.replace(".json",''))
                        bboxes.append([])
                        time_stamps.append(t_stamp)
                        subject_ids.append([])
                        total_index_anno.append(index_anno)
                    else:
                        for ln in range(len(bbox)):
                            anno_id.append(filename.replace(".json",''))
                            time_stamps.append(t_stamp)
                            total_index_anno.append(index_anno)
                        for box in bbox:
                            bboxes.append(box)
                        for s_id in sub_ids:
                            subject_ids.append(s_id)

                out_path=os.path.join(args.output,"{}".format(item))
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                    os.mkdir(os.path.join(out_path,item))
                    
                xlsx_file=os.path.join(out_path,f"{dir_}.xlsx")
                data_dict={'frame_id':anno_id,'bbox':bboxes, 'subject_id':subject_ids,'time_stamps':time_stamps,'indx':total_index_anno}
                writer = pd.ExcelWriter(xlsx_file, engine='openpyxl') 
                wb  = writer.book
                df = pd.DataFrame(data_dict)
                df.to_excel(writer, index=False)
                wb.save(xlsx_file)
                output=str(xlsx_file).split("/")[-1]
                print(f"file {output} is created")
