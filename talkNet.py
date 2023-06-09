import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
import disarray
from sklearn.metrics import confusion_matrix

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()
        self.model = talkNetModel().cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.SGD(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        #self.device=torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            #print(f' audio input size{audioFeature[0].shape}')
            #print(f"labels {labels}")
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)

           # for param in audioEmbed.parameters():
           #     param.requires_grad=False
           # for param in visualEmbed.parameters():
           #     param.requires_grad = False



            #shp=audioEmbed.shape
            #print(audioEmbed.shape)


            outsAV= self.model.forward_audio_visual_backend(audioEmbed,visualEmbed)
            #outsAV=nn.DataParallel(outsAV, device_ids=[1, 3])  # feedForward
            outsA = self.model.forward_audio_backend(audioEmbed)
            #outsA = nn.DataParallel(outsA, device_ids=[1, 3])
            outsV = self.model.forward_visual_backend(visualEmbed)
            #outsV= nn.DataParallel(outsV, device_ids=[1, 3])

            labels = labels[0].reshape((-1)).cuda() # Loss
            #labels=nn.DataParallel(outsV, device_ids=[1, 3])

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            #print(prec.shape)
            #print(prec)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr
    
    def compute_prec_recall(self,pred,g_truth):
        tp=np.where((np.array(pred)==1) & (np.array(g_truth)==1),1,0)
        df = pd.DataFrame (tp.tolist(), columns = ['tp'])
        df['tp'] = df['tp'].cumsum()

        #Compute precision
        precision=df['tp']/(df.index+1)
        df ['precision']= precision

        #Compute recall
        recall=df['tp']/g_truth.count(1)
        df['recall'] = recall

        return (np.array(df['precision']),np.array(df['recall']))
    
    def compute_precision_recall(self, scores, labels, num_gt):
        """Compute precision and recall.

        Args:
          scores: A float numpy array representing detection score
          labels: A boolean numpy array representing true/false positive labels
          num_gt: Number of ground truth instances

        Raises:
          ValueError: if the input is not of the correct format

        Returns:
          precision: Fraction of positive instances over detected ones. This value is
            None if no ground truth labels are present.
          recall: Fraction of detected positive instance over all positive instances.
            This value is None if no ground truth labels are present.

        """
        #if not isinstance(labels, np.ndarray) or labels.dtype != np.bool or len(labels.shape) != 1:
        #    raise ValueError("labels must be single dimension bool numpy array")

        if not isinstance(scores, np.ndarray) or len(scores.shape) != 1:
            raise ValueError("scores must be single dimension numpy array")

        if num_gt  <  np.sum(labels):
            raise ValueError("Number of true positives must be smaller than num_gt.")

        if len(scores) != len(labels):
            raise ValueError("scores and labels must be of the same size.")

        if num_gt == 0:
            return None, None

        sorted_indices = np.argsort(scores)
        sorted_indices = sorted_indices[::-1]
        labels = labels.astype(int)
        true_positive_labels = labels[sorted_indices]
        false_positive_labels = 1 - true_positive_labels
        cum_true_positives = np.cumsum(true_positive_labels)
        cum_false_positives = np.cumsum(false_positive_labels)
        precision = cum_true_positives.astype(float) / (
            cum_true_positives + cum_false_positives)
        recall = cum_true_positives.astype(float) / num_gt
        return precision, recall


    
    def compute_average_precision(self, precision, recall):
        """Compute Average Precision according to the definition in VOCdevkit.
        Precision is modified to ensure that it does not decrease as recall
        decrease.
        Args:
          precision: A float [N, 1] numpy array of precisions
          recall: A float [N, 1] numpy array of recalls
        Raises:
          ValueError: if the input is not of the correct format
        Returns:
          average_precison: The area under the precision recall curve. NaN if
            precision and recall are None.
        """
        if precision is None:
            if recall is not None:
                raise ValueError("If precision is None, recall must also be None")
            return np.NAN

        if not isinstance(precision, np.ndarray) or not isinstance(
                recall, np.ndarray):
            raise ValueError("precision and recall must be numpy array")
        if precision.dtype != np.float or recall.dtype != np.float:
            raise ValueError("input must be float numpy array.")
        if len(precision) != len(recall):
            raise ValueError("precision and recall must be of the same size.")
        if not precision.size:
            return 0.0
        if np.amin(precision) < 0 or np.amax(precision) > 1:
            raise ValueError("Precision must be in the range of [0, 1].")
        if np.amin(recall) < 0 or np.amax(recall) > 1:
            raise ValueError("recall must be in the range of [0, 1].")
        if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
            raise ValueError("recall must be a non-decreasing array")

        recall = np.concatenate([[0], recall, [1]])
        precision = np.concatenate([[0], precision, [0]])

        # Smooth precision to be monotonically decreasing.
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = np.maximum(precision[i], precision[i + 1])

        indices = np.where(recall[1:] != recall[:-1])[0] + 1
        average_precision = np.sum(
            (recall[indices] - recall[indices - 1]) * precision[indices])
        return average_precision
    

    import time

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        predLabels =[]
        ground_truth_Labels= []
        ground_truth_length=0
        count_label=0
        count_time=0
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                tic = time.perf_counter()
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels = labels[0].reshape((-1)).cuda()
                #print((labels.shape))


                _, predScore, predLabel, _ = self.lossAV.forward(outsAV, labels)
                print("shape {shp}".format(shp=labels.size(0)))

                toc = time.perf_counter()

                print(f"time diff {toc - tic:0.4f} seconds")
                dif=toc - tic

                lbl = labels.size(0)

                count_time +=dif
                count_label +=lbl

                predLabel = predLabel.detach().cpu().numpy().astype(int)
                ground_truth_Label =labels.detach().cpu().numpy()
                predLabels.extend(predLabel)
                ground_truth_Labels.extend(ground_truth_Label)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                #print(predScore)
                #print(f"length {len(predScore)}")
                ground_truth_length+=len(predScore)
                
            prec=0
            recall =0
            mAP=0
        print(f"total time:  {count_time:0.4f}")
        print (f"total image: {count_label}")
        acu = balanced_accuracy_score(ground_truth_Labels, predLabels)
        #prec, recall = self.compute_precision_recall(np.array(predScores), np.array(ground_truth_Labels), ground_truth_length)
        #prec, rec= self.compute_prec_recall(predLabels,ground_truth_Labels)
        #print(prec)
            #mAP = self.compute_average_precision(prec,recall)
        cm = confusion_matrix(ground_truth_Labels, predLabels)


        return acu,cm
    def evaluate_network_org(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                labels = labels[0].reshape((-1)).cuda()
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
