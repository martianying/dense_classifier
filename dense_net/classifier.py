
from Dense_helper import *
from dataset import *

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.conv_large = Conv(is_large_feature=True)
        self.conv_small = Conv(is_large_feature=False)
        self.dense_large = DenseNet(init=GROWTH, dense_layer_lst=DENSE_BNUM)
        self.dense_small = DenseNet(init=GROWTH, dense_layer_lst=DENSE_BNUM)
        self.fusion = Fusiontest()
        self.adpt = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(762, cfg.CLAS, kernel_size=(1, 1))

        self.large_dic = {}
        self.small_dic = {}


    def forward(self, x):
        large_f, small_f = x[0], x[1]
        x0 = self.conv_large(large_f)
        x1 = self.conv_small(small_f)
        subs_x0 = self.dense_large(x0)
        subs_x1 = self.dense_small(x1)


        for i in range(len(cfg.DENSE_BNUM)):

            if i not in self.large_dic:
                self.large_dic[i] = subs_x0[i]
                self.small_dic[i] = subs_x1[i]

            else:
                self.large_dic[i] = torch.cat((self.large_dic[i], subs_x0[i]), dim=0)
                self.small_dic[i] = torch.cat((self.small_dic[i], subs_x1[i]), dim=0)

        if self.large_dic[0].shape[0] == cfg.WINDOW_LIMITS:
            subs_x0 = [self.large_dic[i] for i in range(len(cfg.DENSE_BNUM))]
            subs_x1 = [self.small_dic[i] for i in range(len(cfg.DENSE_BNUM))]

            self.large_dic = {}
            self.small_dic = {}

            total = []

            for t in range(len(cfg.DENSE_BNUM)):
                sum_lst = []
                for f in range(cfg.WINDOW_LIMITS):
                    ele = torch.unsqueeze(subs_x0[t][f] + subs_x1[t][f], dim=0)
                    sum_lst.append(ele)
                total.append(sum_lst)

            fmaps = []

            for k in range(len(cfg.DENSE_BNUM)):
                basket = []
                for k1 in range(cfg.WINDOW_LIMITS):
                    basket.append(total[k][k1])
                    ele = torch.cat(tuple(basket), dim=1)
                fmaps.append(ele)


            x = self.fusion(fmaps)
            x = self.adpt(x)
            x = self.conv(x)
            x = x.squeeze()

            return x.unsqueeze(dim=0)



