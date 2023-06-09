import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from focal_loss import  FocalLoss

class lossAV(nn.Module):
    def __init__(self):
        super(lossAV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.FC        = nn.Linear(256, 2)
        self.floss= FocalLoss()
        
    def forward(self, x, labels=None):	
        x = x.squeeze(1)
        x = self.FC(x)
        #print(type(labels))
        if labels == None:
        #if not torch.is_tensor(labels):
            predScore = x[:,1]
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            nloss = self.criterion(x, labels)
            #nloss =self.floss(x,labels)
            predScore = F.softmax(x, dim = -1)
            predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
            #print(predLabel)
            #print(predLabel)
            #predLabel=torch.argmax(x, dim=-1)
            #print(predLabel)
            correctNum = (predLabel == labels).sum().float()
            return nloss, predScore, predLabel, correctNum

class lossA(nn.Module):
    def __init__(self):
        super(lossA, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.floss = FocalLoss()
        self.FC        = nn.Linear(128, 2)

    def forward(self, x, labels):	
        x = x.squeeze(1)
        x = self.FC(x)	
        nloss = self.criterion(x, labels)

        #nloss = self.floss(x,labels)


        return nloss

class lossV(nn.Module):
    def __init__(self):
        super(lossV, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.floss = FocalLoss()
        self.FC        = nn.Linear(128, 2)

    def forward(self, x, labels):	
        x = x.squeeze(1)
        x = self.FC(x)
        nloss = self.criterion(x, labels)
        #nloss = self.floss(x,labels)
        return nloss
