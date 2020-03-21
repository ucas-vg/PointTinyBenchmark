from torch import nn
import torch


class classify_atten(nn.Module):

    def __init__(self, data):
        super(classify_atten, self).__init__()
        self.classes = 200
        self.feat_dim = 2048
        # self.fc = nn.Linear(self.feat_dim, self.classes)
        self.MetaEmbedding_Classifier = MetaEmbedding_Classifier()
        self.criterions = DiscCentroidsLoss(self.classes, self.feat_dim)
        # self.criterions.centroids.data = self.centroids_cal(data)
        self.centroids = torch.zeros(self.classes, self.feat_dim).cuda()
        self.centroids = self.centroids_cal(data).cuda()

    def forward(self, x):
        # _, fea = self.pretrained_model(x)

        # if self.training:
        #    self.centroids = self.criterions.centroids.data
        raw_logits, [values_memory, infused_feature] = self.MetaEmbedding_Classifier(fea, self.centroids)
        return raw_logits, fea, values_memory

    def forward_fea(self, x):
        _, fea = self.pretrained_model(x)
        self.features = fea

    def centroids_cal(self, data):
        centroids = torch.zeros(self.classes, self.feat_dim).cuda()
        print('Calculating centroids.')
        self.pretrained_model.eval().cuda()
        label_count = []
        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for i, data in enumerate(data):
                img, labels = data[0].cuda(), data[1].cuda()
                # Calculate Features of each training data
                self.forward_fea(img)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]
                label_count.extend(labels.cpu().numpy())
        # Average summed features with class count
        centroids /= torch.tensor(class_count(label_count)).float().unsqueeze(1).cuda()
        return centroids
