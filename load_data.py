# coding=utf-8
#This code is part of the Addressee Detection paper
# Copyright (c) 2022 Zhejiang Lab
# Written by Fiseha B. <fisehab@zhejianglab.com>

import numpy as np
import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop


def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        #print(line)
        data = line.split('\t')
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, dataName + '.wav'))
        audioSet[dataName] = audio
        #print(os.path.join(dataPath, dataName + '.wav'))
        #print(audio.shape)
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio

    return audio.astype(numpy.int16)

def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
    dataName = data[0]
    audio = audioSet[dataName]    
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    #print(f"data name{dataName}")
    #print(audio.shape)
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
   ## print(audio.shape)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]


    return audio

def load_visual(data, dataPath, numFrames, visualAug): 
    image_paths=[]
    #print(f"data{data}")
    #for line in data:
    #print(line)
    ##row=data.split("\t")
    row= data
    #print(f"data{data}")
    datapath =dataPath
    if int(row[1])==1:
        image_paths.append(os.path.join(datapath,f"{row[0]}.jpg"))
    else:
        start_frm=round(float(row[0].split("_")[-2])*15)
        end_frm=round(float(row[0].split("_")[-1])*15)
        strt=start_frm
        fl_names=row[0].split(":")
        subj_id= fl_names[1].split("_")[0].replace(".0","")
        while start_frm <=end_frm:
    ##        print(round(start_frm/15,2))
            frm_ts =round(start_frm/15,2)
            fl_name=f"{fl_names[0]}:{subj_id}_{frm_ts}.jpg"
            image_path=os.path.join(datapath,fl_name)
            image_paths.append(image_path)
    ##        print(image_path)
            start_frm+=1
    faces = []
    H = 112
    
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    #print(image_paths[:numFrames])
    for faceFile in image_paths[:numFrames]:
        #print(f"face file{faceFile}")
        face = cv2.imread(faceFile)
        #print(face.shape)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        self.miniBatch.append(sortedMixLst)
        start = 0
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
              break
            start = end

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet))
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, visualAug = False))
            labels.append(load_label(data, numFrames))
            #print(numpy.array(labels))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        ##print('line:{}'.format(line))
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
