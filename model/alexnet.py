import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
import chainer


class MLP(chainer.Chain):
    def __init__(self, n_out, init_net=False):
        super(MLP, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, init_net)

        if init_net:
            caffemodel = ''
            func = caffe.CaffeFunction(caffemodel)
            # Conv1
            self.conv1.W.data = func.conv1.W.data
            self.conv1.b.data = func.conv1.b.data
            # Conv2
            self.conv2.W.data = func.conv2.W.data
            self.conv2.b.data = func.conv2.b.data
            # Conv3
            self.conv3.W.data = func.conv3.W.data
            self.conv3.b.data = func.conv3.b.data
            # Conv4
            self.conv4.W.data = func.conv4.W.data
            self.conv4.b.data = func.conv4.b.data
            # Conv5
            self.conv5.W.data = func.conv5.W.data
            self.conv5.b.data = func.conv5.b.data

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        y = self.fc8(h)
        return y
