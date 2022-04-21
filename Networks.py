from turtle import forward
import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch



class Resnet50_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet50_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method =  nn.AdaptiveMaxPool2d(1)


    def forward(self, input, bb_box = None):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        return F.normalize(x)



class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        #self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        #x = self.pool_method(x).view(-1, 512)
        return x

class Attention(nn.Module):
    def __init__(self,in_chans,out_chans):
        super().__init()
        self.conv1 = nn.Conv2d(in_chans,out_chans,1)
        self.conv2 = nn.Conv2d(in_chans,out_chans,1)

    def forward(self,x):
        x = F.ReLU(self.conv1(x))
        return F.log_softmax(self.conv2(x),dim=-1)

class VGG_Attention_Network(nn.Module):
    def __init__(self,hp):
        super().__init__()
        self.feature_embed = VGG_Network(input=input)
        self.attn = Attention(in_chans=512,out_chans=512)
        self.pool_method = nn.AdaptiveMaxPool2d(1)
    def forward(self,x):
        x = self.feature_embed(x)
        y = self.attn(x)
        z = torch.mul(x,y)
        z = torch.add(z,x)

        z = self.pool_method(z)
        z = z.view(-1,512)
        z = F.normalize(z)

        return z












#Experimenting
    #def Attention(self, feature_embed):
        #n, c, h, w = feature_embed.shape
        #self.conv1 = nn.Conv2d(512,512,1)
        #self.conv2 = nn.Conv2d(512,512,1)

    #def forward(self, input, bb_box = None):
        #x = self.backbone(input)
        #x = self.pool_method(x)
        #y = self.Attention(x)
        #y = F.ReLU(self.conv1(y))
        #y = F.log_softmax(y,dim=1)
        #Connecting Attention mask to the main feature vector and getting the attended feature
        #z = torch.mul(x,y)
        #Shortcut Connection between main feature vector and attended feature vector
        #z = torch.add(z,x)
        #z = self.pool_method(z)
        #z = z.view(-1,512)
        #return F.normalize(z)

class InceptionV3_Network(nn.Module):
    def __init__(self, hp):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default


    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        feature = self.pool_method(backbone_tensor).view(-1, 2048)
        return F.normalize(feature)
