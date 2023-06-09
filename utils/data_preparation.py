import json
import csv
import numpy
import pandas
from tqdm import tqdm
import os
import random
from PIL import Image

## crop face regions
def imcrop(image_path, coords, saved_location):

    li = list(coords.split(", "))
    cord=[]
    for i in li:
        cord.append(int(i.replace('[','').replace(']','')))

    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop((cord[0],cord[1],cord[0]+cord[3],cord[1]+cord[2]))
    cropped_image.save(saved_location)


## read annotaiton file from original Json MUMMER dataset
def read_annoation_file(annot_file):
  
    # Load annotations
    annotations = json.load(open(annot_file))
    annotations = annotations["faces"]
    bbox=[]
    subj_id=[]
    for bb in annotations:
        x = int(bb["x"])
        y = int(bb["y"])
        h = int(bb["h"])
        w = int(bb["w"])
        hid = int(bb["id"])
        bbox.append([x,y,h,w])
        subj_id.append(hid)
    return bbox, subj_id

### Create single CSV data

def create_E_MUMMER_csv_data(input_file,subj):
    fl=input_file
  
    index = fl.index
    cond = fl["subject_id"] == subj
    indexs=index[cond]
    indexs=indexs.tolist()
    indx_len=len(indexs)
    temp_indx= []
    new_indx=[]
    labels=[]           
                
    for indx in range(len(indexs)):
        if(indx!=len(indexs)-1):
            if (fl["indx"][indexs[indx]]==(fl["indx"][indexs[indx+1]])-1):
                temp_indx.append(indexs[indx])
                if(indx==len(indexs)-2):
                    new_indx.append(temp_indx)
            else:
                temp_indx.append(indexs[indx])
                new_indx.append(temp_indx)
                    #print(temp_indx)
                temp_indx= []
        else:
            if fl["indx"][new_indx[-1][-1]]==fl["indx"][indexs[indx]]-1:
                temp_indx.append(indexs[indx])
                new_indx[-1]=temp_indx
            else:
                temp_indx.append(indexs[indx])
                new_indx.append(temp_indx)
                #print(temp_indx)
    return new_indx
      
def split_train_val_test(E_MUMMER,):
    sixty_percent_ds=round(6189*.6)
    twenty_percent_ds= round(6189*.2)
    
    dataset=[]
    datasets=['train.csv','val.csv','test.csv']

    with open(E_MUMMER, "r") as f:
        f = csv.reader(f, delimiter = "\t")
        for line in f:
            dataset.append(line)
    
    random.shuffle(dataset)
    train = dataset[0:sixty_percent_ds]
    print(len(train))
    val=dataset[sixty_percent_ds:sixty_percent_ds+twenty_percent_ds]
    print(len(val))
    test=dataset[sixty_percent_ds+twenty_percent_ds:]
    print(len(val))
    
    for item in datasets:
        if item=="train.csv":
            data=train
        elif item=="val.csv":
            data=val
        else:
            data=test
        out_csv=str(E_MUMMER).replace(str(E_MUMMER).split("/")[-1],item)
        file_=open(out_csv, 'w', newline='\n')
        writer = csv.writer(file_,delimiter="\t")
        for line in data:
            writer.writerow(line)
        print(f"finished writing {item}")
           


        
    
