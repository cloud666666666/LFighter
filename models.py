import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np

class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, return_features=False):
        # 视图1：输入层特征（flatten）
        input_flat = x.view(x.size(0), -1)  # 原始输入视图
        
        # 第一层卷积+激活
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        first_layer_activation = x.view(x.size(0), -1)  # 第一层激活视图
        
        # 继续前向传播
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        logits = self.fc2(x)
        output = F.log_softmax(logits, dim=1)
        
        if return_features:
            # 返回三个视图的特征：输入层、第一层激活、输出层logits
            features = [input_flat, first_layer_activation, logits]
            return features, output
        return output

class BiLSTM(nn.Module):
    def __init__(self, num_words, embedding_dim = 100, dropout = 0.25):
        super(BiLSTM, self).__init__()
        """
        Given embedding_matrix: numpy array with vector for all words
        return prediction ( in torch tensor format)
        """
        self.embedding = nn.Embedding(
                                      num_embeddings=num_words+1,
                                      embedding_dim=embedding_dim)
        # LSTM with hidden_size = 128
        self.lstm = nn.LSTM(
                            embedding_dim, 
                            128,
                            bidirectional=True,
                            batch_first=True,
                             )
        # Input(512) because we use bi-directional LSTM ==> hidden_size*2 + maxpooling **2  = 128*4 = 512, 
        #will be explained more on forward method
        self.out = nn.Linear(512, 1)
    def forward(self, x):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        # fit embedding to LSTM
        self.lstm.flatten_parameters()
        hidden, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool= torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)
        # concat avg_pool and max_pool ( so we have 256 size, also because this is bidirectional ==> 256*2 = 512)
        out = torch.cat((avg_pool, max_pool), 1)
        # fit out to self.out to conduct dimensionality reduction from 512 to 1
        out = self.out(out)
        # return output
        return out
class CNNPATHMNIST(nn.Module):
    def __init__(self):
        super(CNNPATHMNIST, self).__init__()
        # 更保守的架构设计，减少参数数量
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # 3通道输入（RGB）
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 9)
    def forward(self, x, return_features=False):
        input_flat = x.view(x.size(0), -1)  # 原始输入视图（flatten）

        x = F.relu(self.conv1(x))
        first_layer_activation = x.view(x.size(0), -1)  # 第一层激活视图

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)                         # -> 64 x 12 x 12
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        if return_features:
            features = [input_flat, first_layer_activation, logits]
            return features, logits
        return logits


class Block(nn.Module):
    def __init__(self, out_features, nfea, device):
        super(Block, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)

        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)

        self.device = device

    def forward(self, input, adj, view):
        input1 = self.S(self.S_norm(input))
        input2 = self.U(self.U_norm(view))
        output = torch.mm(adj, input)
        output = input1 + input2 - output
        return output

class DBONet(nn.Module):
    def __init__(self, nfeats, n_view, n_clusters, blocks, para, Z_init, device):
        super(DBONet, self).__init__()
        self.n_clusters = n_clusters
        self.blocks = blocks
        self.device = device
        self.n_view = n_view
        self.block = nn.ModuleList([Block(n_clusters, n_feat, device) for n_feat in nfeats])

        self.Z_init = torch.from_numpy(Z_init).float().to(device)
        self.theta = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)

    def soft_threshold(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def forward(self, features, adj):
        output_z = self.Z_init
        for i in range(0, self.blocks):
            z = torch.zeros_like(self.Z_init).to(self.device)
            for j in range(0, self.n_view):
                z += self.block[j](output_z, adj[j], features[j])
            output_z = self.soft_threshold(z / self.n_view)
        return output_z

class RESPATHMNIST(nn.Module):
    def __init__(self, num_classes=9):
        super(RESPATHMNIST, self).__init__()
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=False)
        # 修改输入通道为3（PATHMNIST为3通道）
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改输出类别为9
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

def setup_model(model_architecture, num_classes = None, tokenizer = None, embedding_dim = None, trainset=None):
    available_models = {
        "CNNMNIST": CNNMNIST,
        "CNNPATHMNIST": CNNPATHMNIST,
        "BiLSTM": BiLSTM,
        "ResNet18" : tv.models.resnet18,
        "VGG16" : tv.models.vgg16,
        "DN121": tv.models.densenet121,
        "SHUFFLENET":tv.models.shufflenet_v2_x1_0,
        "DBONet": DBONet,
        "RESPATHMNIST": RESPATHMNIST
    }
    print('--> Creating {} model.....'.format(model_architecture))
    # variables in pre-trained ImageNet models are model-specific.
    if model_architecture == "RESPATHMNIST":
        model = RESPATHMNIST(num_classes=9 if num_classes is None else num_classes)
    elif "ResNet18" in model_architecture:
        model = available_models[model_architecture]()
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, num_classes)
    elif "VGG16" in model_architecture:
        model = available_models[model_architecture]()
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_features, num_classes)
    elif "SHUFFLENET" in model_architecture: 
        model = available_models[model_architecture]()
        model.fc = nn.Linear(1024, num_classes)
    elif 'BiLSTM' in model_architecture:
        if tokenizer is None:
            raise ValueError("tokenizer 不能为 None，请确保传入了 tokenizer 参数。")
        model = available_models[model_architecture](num_words =  len(tokenizer.word_index), embedding_dim = embedding_dim)
    elif model_architecture == 'DBONet':
        # PATHMNIST三视图：输入层2352，第一层激活25088，输出层9
        nfeats = [2352, 25088, 9]
        n_view = 3
        n_clusters = 9
        blocks = 2
        para = 0.1
        num_samples = len(trainset) if trainset is not None else 10000
        np.random.seed(42)  # 保证Z_init可复现
        Z_init = np.random.randn(num_samples, n_clusters)
        import config
        device = config.DEVICE
        model = DBONet(nfeats, n_view, n_clusters, blocks, para, Z_init, device)
    else:
        model = available_models[model_architecture]()

    if model is None:
        print("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)
    print('--> Model has been created!')
    return model