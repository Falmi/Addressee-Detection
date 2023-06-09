
import time, os, torch, argparse, warnings, glob
from load_data import train_loader, val_loader
from utils.tools import *
from talkNet import talkNet
import torch.nn as nn
import numpy as np
import pandas as pd

def main():
    #os.environ['CUDA_VISIBLE_DEVICES']="0"
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    #parser.add_argument('--batchSize',    type=int,   default=2500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--batchSize',    type=int,   default=100,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    #parser.add_argument('--dataPathAVA',  type=str, default="/data08/AVA", help='Save path of AVA dataset')
    parser.add_argument('--dataPathAVA',  type=str, default="./data", help='Save path of E_MUMMER dataset')
    parser.add_argument('--savePath',     type=str, default="exps/Abilation_wo_SA_") # Data selection #SA_CA_CBP
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation onlyexp_apha_no_pretrained_CE_newattention
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation', type=str, default='false', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    #parser.add_argument('--pretrained',      type=str, default="ava_pretrained", help='initialize from AVA pretrain model')
    parser.add_argument('--pretrained', type=str, default="False", help='initialize from AVA pretrain model')

    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    loader = train_loader(trialFileName = args.trainTrialAVA, \
                          audioPath      = args.audioPathAVA, \
                          visualPath     = args.visualPathAVA, \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPath     = args.audioPathAVA, \
                        visualPath    = args.visualPathAVA, \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)

    if args.evaluation == "eval":
        #download_pretrain_model_AVA()
        s = talkNet(**vars(args))
        path_Eval= os.path.join(args.modelSavePath,"model_0022.model")
        s.loadParameters(path_Eval)
        print("Model %s loaded from previous state!"%(path_Eval))
        acu,cm= s.evaluate_network(loader = valLoader, **vars(args))
        #print("mAP %2.2f%%"%(mAP))
        print("Balaced accuracy %2.2f"%(acu))

        df = pd.DataFrame(cm, index=['Speak to robot', 'Speak to another Subject'],
                          columns=['Speak to robot', 'Speak to another Subject'])
        print(df.da.export_metrics())
        #print("precision and recall %2.2f %2.2f"%(prec, rec))
        #print("Mean average precision %2.2f"%(mAP))
        quit()

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = talkNet(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    elif (args.pretrained =="ava_pretrained"):
        print(" load from AVA pretrained model")
        class Indetity(nn.Module):
            def __init__(self):
                super(Indetity, self).__init__()
            def forward(self,x):
                return x
        
        epoch = 1
        s = talkNet(epoch = epoch, **vars(args))
        s.loadParameters('pretrain_AVA.model')
        #s.lossA.FC=nn.Linear(in_features=128, out_features=4, bias=True).cuda()
        #s.lossV.FC=nn.Linear(in_features=128, out_features=4, bias=True).cuda()
        #s.lossAV.FC=nn.Linear(in_features=256, out_features=4, bias=True).cuda()

        #print(s.lossA.FC)
        #print(s.lossV.FC)
        #print(s.lossAV.FC)
    else:
        epoch = 1
        s = talkNet(epoch = epoch, **vars(args))

    Acus = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            accu,cm =s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args))
            Acus.append(accu)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, Acu %2.2f%%, bestAcu %2.2f%%"%(epoch, Acus[-1], max(Acus)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, Acu %2.2f%%, bestAcu %2.2f%%\n"%(epoch, lr, loss, Acus[-1], max(Acus)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1
        

if __name__ == '__main__':
    main()
