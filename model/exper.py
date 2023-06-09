from torch import nn
from torch.autograd import Variable
from model.compact_bilinear_pooling import CompactBilinearPooling

bottom1 = Variable(torch.randn(128, 512, 14, 14)).cuda()
bottom2 = Variable(torch.randn(128, 512, 14, 14)).cuda()

layer = CompactBilinearPooling(512, 512, 8000)
layer.cuda()
layer.train()

out = layer(bottom1, bottom2)