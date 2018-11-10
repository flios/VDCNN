import torch
from torch import nn
from torch.autograd import Variable



class KMaxPool(nn.Module):

    def __init__(self, k = None):
        super().__init__()
        self.k = k

    def forward(self, x):
        if self.k is None:
            time_steps = x.shape[2]
            self.k = time_steps//2
        kmax, kargmax = x.topk(self.k, dim=2)
        return kmax

def downsampling(pool_type = 'resnet', channel=None):
    if pool_type == 'resnet':
        pool = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel))
    elif pool_type == 'kmaxpool':
        pool = KMaxPool()
    elif pool_type == 'vgg':
        pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    else:
        pool = None
    return pool

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs ):
        super().__init__()
        if kwargs is not None:
            kernel_size = kwargs.pop('kernel_size',3)
            downsample = kwargs.pop('downsample', None)
            optional_shortcut = kwargs.pop('optional_shortcut', False)
            shortcut = kwargs.pop('shortcut', None)


        padding = kernel_size//2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsampling(downsample,out_channels)
        self.shortcut = shortcut
        # None if downsample is None else nn.Sequential(
        #     nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2),
        #     nn.BatchNorm1d(out_channels))
        self.optional_shortcut = optional_shortcut


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.optional_shortcut:
            if self.downsample is not None:
                # print(out.shape,residual.shape)
                # print(out.shape, residual.shape)
                out = self.downsample(out)
                # print(out.shape)
            if self.shortcut is not None:
                residual = self.shortcut(residual)
            out += residual
            out = self.relu2(out)
        else:
            out = self.relu2(out)
            if self.downsample is not None:
                out = self.downsample(out)

        return out



class VDCNN(nn.Module):
    def __init__(self, block, layers, n_classes, vocabulary_size, **kwargs):
        super().__init__()

        self.inplanes = 64

        if kwargs is not None:
            kernel_size = kwargs.pop('kernel_size',3)
            downsample = kwargs.pop('downsample','resnet')
            optional_shortcut = kwargs.pop('optional_shortcut', False)
            kmax = kwargs.pop('kmax',8)

        layer_kwargs = {
            'kernel_size' : kernel_size,
            'downsample' : downsample,
            'optional_shortcut' : optional_shortcut
            }



        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16, padding_idx=0)
        self.conv1 = nn.Conv1d(16, 64, kernel_size=3, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], **layer_kwargs)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], **layer_kwargs)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], **layer_kwargs)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], **layer_kwargs)

        self.kmax_pooling = KMaxPool(k=kmax)

        fc = []
        fc.append(nn.Linear(512 * kmax, 2048))
        fc.append(nn.Linear(2048, 2048))
        fc.append(nn.Linear(2048, n_classes))

        self.fc = nn.Sequential(*fc)


    def _make_layer(self, block, channels, num_block, **kwargs):
        if kwargs is not None:
            kernel_size = kwargs.pop('kernel_size',3)
            downsample = kwargs.pop('downsample','resnet')
            optional_shortcut = kwargs.pop('optional_shortcut', False)

        layers = []
        for _ in range(num_block-1):
            shortcut =None
            if self.inplanes != channels:
                shortcut =nn.Sequential(
                    nn.Conv1d(self.inplanes, channels, kernel_size=1, stride=1),
                    nn.BatchNorm1d(channels))

            layers.append(block(self.inplanes, channels, kernel_size=kernel_size, optional_shortcut=optional_shortcut, shortcut=shortcut))
            self.inplanes = channels

        shortcut = nn.Sequential(
            nn.Conv1d(self.inplanes, channels, kernel_size=1, stride=2),
            nn.BatchNorm1d(channels))
        layers.append(block(self.inplanes, channels,
            optional_shortcut=optional_shortcut, downsample=downsample, kernel_size=kernel_size, shortcut = shortcut))
        self.inplanes = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.kmax_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def vdcnn9(n_classes, vocabulary_size, **kwargs):
    model = VDCNN(BasicBlock,[1,1,1,1],n_classes,vocabulary_size,**kwargs)
    return model

def vdcnn17(n_classes, vocabulary_size, **kwargs):
    model = VDCNN(BasicBlock,[2,2,2,2],n_classes,vocabulary_size,**kwargs)
    return model

def vdcnn29(n_classes, vocabulary_size, **kwargs):
    model = VDCNN(BasicBlock,[5,5,2,2],n_classes,vocabulary_size,**kwargs)
    return model

def vdcnn49(n_classes, vocabulary_size, **kwargs):
    model = VDCNN(BasicBlock,[8,8,5,3],n_classes,vocabulary_size,**kwargs)
    return model



if __name__ == '__main__':

    model = vdcnn49(vocabulary_size=29, embed_size=16, n_classes=2, k=2, optional_shortcut=True)
    rand_inputs = Variable(torch.LongTensor(8, 1104).random_(0, 29))
    print(model(rand_inputs))
