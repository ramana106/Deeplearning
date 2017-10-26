import chainer.functions as F
import chainer.links as L
import chainer


class MLP(chainer.Chain):
    def __init__(self, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l6 = L.Linear(None, 1024)
            self.l7 = L.Linear(None, 1024)
            self.l8 = L.Linear(None, 256)
            self.l9 = L.Linear(None, n_out)

    # def __call__(self, x):
    #     h = F.dropout(F.relu(self.l6(x)))
    #     h = F.dropout(F.relu(self.l7(h)))
    #     h = F.dropout(F.relu(self.l8(h)))
    #     y = self.l9(h)
    #     return y

    def __call__(self, x):
        h = F.relu(self.l6(x))
        h = F.relu(self.l7(h))
        h = F.relu(self.l8(h))
        y = self.l9(h)
        return y
