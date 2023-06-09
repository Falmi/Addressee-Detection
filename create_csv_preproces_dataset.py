# coding=utf-8
#This code is part of the Addressee Detection paper
# Copyright (c) 2022 Zhejiang Lab
# Written by Fiseha B. <fisehab@zhejianglab.com>

import pandas as pd
import numpy as np
import csv
from pathlib import Path
import argparse
import os
import sys
from utils.data_preparation import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create csv fil annotation file for all xlsx files")
    #input_Path="../../MUMMER/MuMMER/annotations"
    parser.add_argument(
        "--input", type=str, required=True,
        help="A directory of annotation  files")
    parser.add_argument(
        "--output", type=str, default="./data/E_MUMMER_csv/",
        help="out put directory")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    
    assert os.path.isdir(args.input), "Argument '--input' should be a directory name"
    assert os.path.isdir(args.input), "Argument '--output' should be a directory name"

    
    output_path=args.output   
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        # create csv file
    out_put_csv=os.path.join(output_path,"all_E_MUMMER_dataset.csv")
    if os.path.isfile(out_put_csv):
        os.remove(out_put_csv)
    out_put_csv=os.path.join(output_path,"all_E_MUMMER_dataset.csv")
    file_=open(out_put_csv, 'a', newline='\n')
    writer = csv.writer(file_,delimiter="\t")

    sequ_index=0
    anno_path="./data/E_MUMMER_anno_org/"
    folder = ['kinect','intel']
    for fld in folder:
        full_anno_path = os.path.join(anno_path,fld)
        for file in Path(full_anno_path).glob("*.xlsx"):

                #exit();
            ####remove empty labels########
            print(file)
            xls_file=pd.read_excel(file,engine='openpyxl')
            xls_file['Label'].replace(' ', np.nan, inplace=True)
            xls_file.dropna(subset=['Label'], inplace=True)
            ### remove empty Subjects
            xls_file['subject_id'].replace('[]', np.nan, inplace=True)
            xls_file.dropna(subset=['subject_id'], inplace=True)
            ####remove invlaid label, which is 7, if it exists
            xls_file['Label'].replace(7, np.nan, inplace=True)
            xls_file.dropna(subset=['Label'], inplace=True)

                   #### adjust labels
            xls_file['Label'].replace(2, 0, inplace=True)
            xls_file['Label'].replace(3, 1, inplace=True)
            xls_file['Label'].replace(6, 2, inplace=True)
            xls_file['Label'].replace(4, 3, inplace=True)


            ## remove robot speaking frmaes
            #return indices of label 5
            index = xls_file.index
            condition = xls_file["Label"] == 5
            frame_indices = index[condition]
            frame_indices_list =frame_indices.tolist()
            all_image_indices=[]
            count=1
            indexs = []
            for indx in frame_indices_list:
                condition = xls_file["frame_id"] == xls_file["frame_id"][indx]
                if count==1:
                    indexs=xls_file[condition].index
                    count=2
                else:
                    indexs=indexs.append(xls_file[condition].index)
            xls_file.drop(indexs, inplace=True)

            ## write to csv file
            frame_name=str(file).split("/")[-1].split(".")[0]

            subjects=xls_file["subject_id"].unique()
        #print(subjects)
            #print(file)
            for subj in subjects:
                new_indx=create_E_MUMMER_csv_data(xls_file,int(subj))
                for j in new_indx:
                    sequ_index+=1
                    label=[]
                    for k in j:
                        label.append(int(xls_file["Label"][k]))
                #labels.append(label)
                    strt=str(round(xls_file["time_stamps"][j[0]], 2))
                    end=str(round(xls_file["time_stamps"][j[-1]],2))
                    if (len(label)>1):
                        fl_nm=f"{fld}_{frame_name}:{str(subj)}_{strt}_{end}"
                    else:
                        fl_nm=f"{fld}_{frame_name}:{str(subj)}_{strt}"
                    writer.writerow([fl_nm,len(label),15.0,label,sequ_index])
                print(f"{fld}/{frame_name} subject {subj} is writen to csv file")
    
    #split the dataset into training, validation and testing
    split_train_val_test(out_put_csv)
