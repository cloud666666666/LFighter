import copy
import enum
from sklearn.decomposition import PCA
import torch
import numpy as np
import math
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
import hdbscan
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from utils import *
from sklearn.preprocessing import StandardScaler
def get_pca(data, threshold = 0.99):
    normalized_data = StandardScaler().fit_transform(data)
    pca = PCA()
    reduced_data = pca.fit_transform(normalized_data)
    # Determine explained variance using explained_variance_ration_ attribute
    exp_var = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var)
    select_pcas = np.where(cum_sum_eigenvalues <=threshold)[0]
    # print('Number of components with variance <= {:0.0f}%: {}'.format(threshold*100, len(select_pcas)))
    reduced_data = reduced_data[:, select_pcas]
    return reduced_data
eps = np.finfo(float).eps
class LFD():
    def __init__(self, num_classes):
        self.memory = np.zeros([num_classes])
    
    def clusters_dissimilarity(self, clusters):
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1 
        cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
        cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
        mincs0 = np.min(cs0, axis=1)
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0/m * (1 - np.mean(mincs0))
        ds1 = n1/m * (1 - np.mean(mincs1))
        return ds0, ds1

    def aggregate(self, global_model, local_models, ptypes):
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        dw = [None for i in range(m)]
        db = [None for i in range(m)]
        for i in range(m):
            dw[i]= global_model[-2].cpu().data.numpy() - \
                local_models[i][-2].cpu().data.numpy() 
            db[i]= global_model[-1].cpu().data.numpy() - \
                local_models[i][-1].cpu().data.numpy()
        dw = np.asarray(dw)
        db = np.asarray(db)

        "If one class or two classes classification model"
        if len(db[0]) <= 2:
            data = []
            for i in range(m):
                data.append(dw[i].reshape(-1))
        
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            labels = kmeans.labels_

            clusters = {0:[], 1:[]}
            for i, l in enumerate(labels):
                clusters[l].append(data[i])

            good_cl = 0
            cs0, cs1 = self.clusters_dissimilarity(clusters)
            if cs0 < cs1:
                good_cl = 1

            # print('Cluster 0 weighted variance', cs0)
            # print('Cluster 1 weighted variance', cs1)
            # print('Potential good cluster is:', good_cl)
            scores = np.ones([m])
            for i, l in enumerate(labels):
                # print(ptypes[i], 'Cluster:', l)
                if l != good_cl:
                    scores[i] = 0
                
            global_weights = average_weights(local_weights, scores)
            return global_weights

        "For multiclassification models"
        norms = np.linalg.norm(dw, axis = -1) 
        self.memory = np.sum(norms, axis = 0)
        self.memory +=np.sum(abs(db), axis = 0)
        max_two_freq_classes = self.memory.argsort()[-2:]
        print('Potential source and target classes:', max_two_freq_classes)
        data = []
        for i in range(m):
            data.append(dw[i][max_two_freq_classes].reshape(-1))

        # === 统一降维处理：确保消融实验公平性 ===
        def unified_dimension_reduction(features, target_dim=200, method='auto', standardize=True):
            """统一降维函数：与多视图算法保持一致"""
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            # 1. 标准化处理
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
                std_info = "standardized"
            else:
                std_info = "raw"
            
            # 2. 智能降维策略
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim <= effective_target_dim:
                return features_array, f"keep_all_{original_dim}_{std_info}"
            elif original_dim <= effective_target_dim * 2:
                return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=effective_target_dim, random_state=42)
                reduced_features = pca.fit_transform(features_array)
                explained_ratio = np.sum(pca.explained_variance_ratio_)
                return reduced_features, f"pca_{effective_target_dim}_var{explained_ratio:.3f}_{std_info}"
        
        # 应用统一降维（与多视图算法保持一致）
        data_reduced, reduction_method = unified_dimension_reduction(data, target_dim=200)
        print(f'[LFighter] Applied unified dimension reduction: {reduction_method}')
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data_reduced)
        labels = kmeans.labels_

        clusters = {0:[], 1:[]}
        for i, l in enumerate(labels):
          clusters[l].append(data_reduced[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1

        # print('Cluster 0 weighted variance', cs0)
        # print('Cluster 1 weighted variance', cs1)
        # print('Potential good cluster is:', good_cl)
        scores = np.ones([m])
        for i, l in enumerate(labels):
            # print(ptypes[i], 'Cluster:', l)
            if l != good_cl:
                scores[i] = 0
            
        global_weights = average_weights(local_weights, scores)
        return global_weights

################################################
# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
       
    def score_gradients(self, local_grads, selectec_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selectec_peers]+= grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]


#######################################################################################
class LFighterDBO:
    def __init__(self):
        pass
    
    def aggregate(self, simulation_model, local_weights, local_features=None):
        import config
        device = simulation_model.device if hasattr(simulation_model, 'device') else config.DEVICE
        
        import torch
        from models import DBONet
        from sklearn.neighbors import kneighbors_graph
        
        # 直接在这里定义归一化函数，避免循环导入
        def normalization(data):
            """数据归一化函数"""
            maxVal = torch.max(data)
            minVal = torch.min(data)
            data = (data - minVal) / (maxVal - minVal + 1e-10)  # 避免除零
            return data

        def standardization(data):
            """数据标准化函数"""
            rowSum = torch.sqrt(torch.sum(data**2, 1))
            repMat = rowSum.repeat((data.shape[1], 1)) + 1e-10
            data = torch.div(data, repMat.t())
            return data
        
        m = len(local_weights)
        print(f"[LFighter-DBO] Processing {m} clients with lightweight DBONet")
        
        # 只提取输出层梯度特征
        local_models = []
        for local_weight in local_weights:
            model = copy.deepcopy(simulation_model)
            model.load_state_dict(local_weight)
            local_models.append(list(model.parameters()))
        
        global_model = list(simulation_model.parameters())
        
        # 使用与LFighter一致的关键类别检测和提取
        # 首先检测最异常的两个类别（与LFighter逻辑完全一致）
        memory = np.zeros(global_model[-1].shape[0])  # 输出层偏置的维度
        
        for i in range(m):
            # 计算权重和偏置的差异
            dw = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            db = global_model[-1].cpu().data.numpy() - local_models[i][-1].cpu().data.numpy()
            
            # 累积异常程度（与原版LFighter的逻辑一致）
            norms = np.linalg.norm(dw, axis=-1)
            memory += norms + np.abs(db)
        
        # 找到最异常的两个类别
        max_two_freq_classes = memory.argsort()[-2:]
        print(f'[LFighter-DBO] Detected anomalous classes (same as LFighter): {max_two_freq_classes}')
        
        # 只提取这两个类别的输出层梯度（与LFighter完全一致）
        gradients = []
        for i in range(m):
            dw = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            key_classes_grad = dw[max_two_freq_classes].reshape(-1)
            gradients.append(key_classes_grad)
        
        feature_matrix = np.array(gradients)
        print(f"[LFighter-DBO] Gradient matrix shape: {feature_matrix.shape}")
        
        # === 统一降维处理：确保消融实验公平性 ===
        def unified_dimension_reduction(features, target_dim=200, method='auto', standardize=True):
            """统一降维函数：与多视图算法保持一致"""
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            # 1. 标准化处理
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
                std_info = "standardized"
            else:
                std_info = "raw"
            
            # 2. 智能降维策略
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim <= effective_target_dim:
                return features_array, f"keep_all_{original_dim}_{std_info}"
            elif original_dim <= effective_target_dim * 2:
                return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=effective_target_dim, random_state=42)
                reduced_features = pca.fit_transform(features_array)
                explained_ratio = np.sum(pca.explained_variance_ratio_)
                return reduced_features, f"pca_{effective_target_dim}_var{explained_ratio:.3f}_{std_info}"
        
        # 应用统一降维（与多视图算法保持一致）
        feature_matrix_reduced, reduction_method = unified_dimension_reduction(feature_matrix, target_dim=200)
        print(f"[LFighter-DBO] Applied unified dimension reduction: {reduction_method}")
        
        # 轻量级DBONet配置：blocks=1-2，快速训练
        n_view = 1
        nfeats = [feature_matrix_reduced.shape[1]]  # 使用降维后的维度
        n_clusters = 2
        blocks = 2  # 统一配置
        para = 0.05  # 统一参数（与MV-DBO保持一致）
        np.random.seed(42)
        Z_init = np.random.randn(m, n_clusters) * 0.01  # 统一初始化缩放
        
        # 统一邻接矩阵配置
        n_neighbors = min(5, m-1)  # 与MV-DBO保持一致的邻居数量
        adj = kneighbors_graph(feature_matrix_reduced, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
        adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
        
        # 创建轻量级DBONet
        dbo_model = DBONet(nfeats, n_view, n_clusters, blocks, para, Z_init, device)
        features_tensor = [torch.tensor(feature_matrix_reduced, dtype=torch.float32, device=device)]
        adjs = [adj_tensor]
        
        # 统一归一化流程（与MV-DBO一致）
        features_norm = []
        for i in range(n_view):
            # 多步归一化
            feature = features_tensor[i]
            feature = (feature - feature.mean(dim=0)) / (feature.std(dim=0) + 1e-8)  # 标准化
            feature = standardization(normalization(feature))  # 进一步归一化
            features_norm.append(feature)
        
        # 统一训练配置：与MV-DBO完全相同
        dbo_model.train()
        optimizer = torch.optim.Adam(dbo_model.parameters(), lr=5e-4, weight_decay=1e-5)  # 统一优化器配置
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)  # 统一调度器
        criterion = torch.nn.MSELoss()
        
        best_loss = float('inf')
        for epoch in range(8):  # 统一训练轮数：8个epoch
            optimizer.zero_grad()
            output_z = dbo_model(features_tensor, adjs)
            
            # 统一损失函数：显式+隐式损失（与MV-DBO一致）
            loss_dis = torch.tensor(0., device=device)
            loss_lap = torch.tensor(0., device=device)
            
            for k in range(n_view):
                # 显式损失：特征重构
                target_sim = features_norm[k] @ features_norm[k].t()
                pred_sim = output_z @ output_z.t()
                loss_dis += criterion(pred_sim, target_sim)
                
                # 隐式损失：图拉普拉斯正则化
                loss_lap += criterion(pred_sim, adjs[k])
            
            # 统一总损失权重
            total_loss = loss_dis + 0.3 * loss_lap
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(dbo_model.parameters(), max_norm=1.0)  # 统一梯度裁剪
            optimizer.step()
            scheduler.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            
            if epoch % 2 == 0:
                print(f"[LFighter-DBO] Epoch {epoch+1}: Loss={total_loss.item():.6f}")
        
        # === 使用LFD clusters_dissimilarity替换Silhouette评估 ===
        dbo_model.eval()
        with torch.no_grad():
            output_z = dbo_model(features_tensor, adjs)
        
        z_np = output_z.detach().cpu().numpy()
        
        # 简化的kmeans聚类（与LFighter-MV-DBO一致）
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(z_np)
        labels = kmeans.labels_
        
        # === 使用LFD的clusters_dissimilarity进行质量评估 ===
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(z_np[i])
        
        # 创建LFD实例并调用clusters_dissimilarity方法
        lfd = LFD(config.NUM_CLASSES)
        cs0, cs1 = lfd.clusters_dissimilarity(clusters)
        good_cl = 0 if cs0 < cs1 else 1
        
        print(f"[LFighter-DBO] LFD cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}")
        
        # 权重分配
        scores = np.ones(m)
        for i, l in enumerate(labels):
            if l != good_cl:
                scores[i] = 0
        
        print(f"[LFighter-DBO] Good clients: {np.sum(scores)}/{m}")
        return average_weights(local_weights, scores)


class LFighterMV:
    def __init__(self):
        pass
    
    def aggregate(self, simulation_model, local_weights, local_features, local_models, peers_types, lfd):
        """LFighter-MV: 三视图拼接 - 输出层梯度，第一层激活值，输入层梯度"""
        # 严格检查：缺少特征时直接报错
        if not local_features or len(local_features) == 0:
            raise ValueError("LFighter-MV requires local_features but got None or empty list. Please check if 'mv' rule is properly extracting features during training.")
        
        m = len(local_weights)
        
        # 视图1: 输出层梯度（权重差异）- 与原版LFighter保持一致
        output_grad_features = []
        
        # 首先计算所有类别的梯度差异来检测最异常的两个类别（与LFighter保持一致）
        global_params = list(simulation_model.parameters())
        
        # 计算所有客户端的输出层偏置差异，用于检测异常类别
        memory = np.zeros(global_params[-1].shape[0])  # 输出层偏置的维度
        
        for i in range(m):
            local_model = copy.deepcopy(simulation_model)
            local_model.load_state_dict(local_weights[i])
            local_params = list(local_model.parameters())
            
            # 计算权重和偏置的差异
            dw = global_params[-2].cpu().data.numpy() - local_params[-2].cpu().data.numpy()
            db = global_params[-1].cpu().data.numpy() - local_params[-1].cpu().data.numpy()
            
            # 累积异常程度（与原版LFighter的逻辑一致）
            norms = np.linalg.norm(dw, axis=-1)
            memory += norms + np.abs(db)
        
        # 找到最异常的两个类别（与原版LFighter一致）
        max_two_freq_classes = memory.argsort()[-2:]
        print(f'[LFighter-MV] Detected anomalous classes (same as LFighter): {max_two_freq_classes}')
        
        # 现在只提取这两个类别的输出层梯度
        for i in range(m):
            local_model = copy.deepcopy(simulation_model)
            local_model.load_state_dict(local_weights[i])
            local_params = list(local_model.parameters())
            
            # 只提取最异常两个类别的权重差异（与LFighter完全一致）
            output_grad = global_params[-2].cpu().data.numpy() - local_params[-2].cpu().data.numpy()
            key_classes_grad = output_grad[max_two_freq_classes].reshape(-1)
            output_grad_features.append(key_classes_grad)
        
        # 视图2: 第一层激活值（从local_features中提取）
        first_activation_features = []
        for i, peer_features in enumerate(local_features):
            if not peer_features or len(peer_features) == 0:
                raise ValueError(f"Empty peer_features for peer {i}. Check feature extraction during training.")
            
            # 调试：打印实际的数据结构
            print(f"[Debug] Peer {i} features structure: type={type(peer_features)}, len={len(peer_features) if hasattr(peer_features, '__len__') else 'N/A'}")
            if hasattr(peer_features, '__len__') and len(peer_features) > 0:
                print(f"[Debug] First element type: {type(peer_features[0])}")
                if hasattr(peer_features[0], '__len__'):
                    print(f"[Debug] First element length: {len(peer_features[0])}")
            
            # 处理可能的嵌套结构
            # peer_features应该是[input_flat, first_layer_activation, logits]
            # 但如果是嵌套的，可能需要额外处理
            actual_features = peer_features
            
            # 检查是否是嵌套结构（list of lists）
            if isinstance(peer_features, (list, tuple)) and len(peer_features) > 0:
                if isinstance(peer_features[0], (list, tuple)):
                    # 如果是嵌套的，取第一个元素（假设它是我们需要的特征）
                    actual_features = peer_features[0]
                    print(f"[Debug] Detected nested structure, using first element")
            
            # 现在检查actual_features是否符合预期格式
            if not isinstance(actual_features, (list, tuple)) or len(actual_features) < 2:
                raise ValueError(f"Invalid features structure for peer {i}: expected list/tuple with >=2 elements, got {type(actual_features)} with length {len(actual_features) if hasattr(actual_features, '__len__') else 'unknown'}")
            
            # 从CNNPATHMNIST的return_features中提取第一层激活值（索引1）
            first_activation = actual_features[1]  # first_layer_activation
            print(f"[Debug] Peer {i} first_activation type: {type(first_activation)}")
            
            if not hasattr(first_activation, 'detach'):
                # 如果first_activation不是tensor，尝试转换
                if isinstance(first_activation, (list, tuple)):
                    # 如果是list/tuple，可能包含多个tensor，取第一个
                    if len(first_activation) > 0 and hasattr(first_activation[0], 'detach'):
                        first_activation = first_activation[0]
                        print(f"[Debug] Converted from list to tensor for peer {i}")
                    else:
                        raise ValueError(f"Cannot convert first_activation to tensor for peer {i}: {type(first_activation)}")
                else:
                    raise ValueError(f"Expected tensor with .detach() method for peer {i}, got {type(first_activation)}")
            
            # 只取第一个样本的激活值作为代表
            first_activation_flat = first_activation[0].detach().cpu().numpy().flatten()
            first_activation_features.append(first_activation_flat)
            print(f"[Debug] Successfully extracted features for peer {i}, shape: {first_activation_flat.shape}")
        
        # 视图3: 输入层梯度（第一层卷积权重的梯度）
        input_grad_features = []
        for i in range(m):
            # 获取第一层卷积层的权重梯度（conv1权重）
            input_grad = list(simulation_model.parameters())[0].cpu().data.numpy() - list(local_models[i].parameters())[0].cpu().data.numpy()
            input_grad_features.append(input_grad.reshape(-1))
        
        # 智能维度处理：根据特征维度选择裁剪或PCA降维策略
        def smart_dimension_reduction(features, target_dim=200, method='auto', standardize=True):
            """智能维度缩减：标准化 + PCA/裁剪，考虑样本数限制"""
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            # 1. 强烈建议的标准化处理（均值0，方差1）
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
                std_info = "standardized"
            else:
                std_info = "raw"
            
            # 2. PCA最大维度不能超过min(样本数, 特征数)
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim <= effective_target_dim:
                # 维度已经足够小，直接返回标准化后的特征
                return features_array, f"keep_all_{original_dim}_{std_info}"
            elif original_dim <= effective_target_dim * 2:
                # 维度适中，使用简单裁剪（速度优先）
                return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
            else:
                # 维度很大，使用PCA降维（信息保留优先）
                if method == 'pca' or method == 'auto':
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=effective_target_dim, random_state=42)
                    reduced_features = pca.fit_transform(features_array)
                    explained_ratio = np.sum(pca.explained_variance_ratio_)
                    return reduced_features, f"pca_{effective_target_dim}_var{explained_ratio:.3f}_{std_info}"
                else:
                    return features_array[:, :effective_target_dim], f"truncate_{effective_target_dim}_{std_info}"
        
        # 对三个视图分别进行智能降维
        output_reduced, output_method = smart_dimension_reduction(output_grad_features, target_dim=200)
        activation_reduced, activation_method = smart_dimension_reduction(first_activation_features, target_dim=200)
        input_reduced, input_method = smart_dimension_reduction(input_grad_features, target_dim=200)
        
        # 自动加权：基于聚类分离度计算每个视图的贡献权重
        def compute_view_weights(view_features_list, view_names):
            """基于标签翻转检测理论计算视图权重 - 输出层应占主导地位"""
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans
            
            weights = []
            scores = []
            
            # 定义基于理论的基础权重（输出层最重要）
            theory_based_weights = {
                'Output_Grad': 0.9,      # 输出层梯度最重要 - 直接反映标签翻转
                'First_Activation': 0.05, # 激活值特征次要 - 反映中间表示变化  
                'Input_Grad': 0.05       # 输入层梯度最低 - 距离攻击最远
            }
            
            for i, (view_features, view_name) in enumerate(zip(view_features_list, view_names)):
                # 获取理论权重作为基准
                base_weight = theory_based_weights.get(view_name, 0.33)
                
                # 对每个视图单独做聚类评估
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(view_features)
                labels = kmeans.labels_
                
                # 检查聚类是否有效（两个簇都有样本）
                if len(set(labels)) > 1:
                    # 计算Silhouette分数
                    sil_score = silhouette_score(view_features, labels)
                    # 计算聚类分离度（簇间距离/簇内距离）
                    centers = kmeans.cluster_centers_
                    inter_dist = np.linalg.norm(centers[0] - centers[1])
                    intra_dist = np.mean([np.std(view_features[labels == k], axis=0).mean() for k in range(2)])
                    separation_ratio = inter_dist / (intra_dist + 1e-8)
                    
                    # 聚类质量分数 (0-1范围)
                    cluster_quality = max(0, sil_score) + min(1.0, np.log(1 + separation_ratio) * 0.1)
                    cluster_quality = min(1.0, cluster_quality)  # 限制在[0,1]
                else:
                    cluster_quality = 0.0
                    
                # 结合理论权重和聚类质量：理论权重70%，聚类质量30%
                # 这确保输出层始终保持较高权重，同时考虑实际数据的聚类效果
                combined_score = base_weight * 0.7 + cluster_quality * base_weight * 0.3
                scores.append(combined_score)
            
            # 权重归一化
            scores = np.array(scores)
            weights = scores / np.sum(scores)  # 归一化
            
            return weights, scores
        
        # 计算三个视图的自适应权重
        view_features_list = [output_reduced, activation_reduced, input_reduced]
        view_names = ['Output_Grad', 'First_Activation', 'Input_Grad']
        view_weights, view_scores = compute_view_weights(view_features_list, view_names)
        
        # 加权拼接降维后的三个视图
        fused_features = []
        for i in range(m):
            # 应用自适应权重
            weighted_output = output_reduced[i] * view_weights[0]
            weighted_activation = activation_reduced[i] * view_weights[1] 
            weighted_input = input_reduced[i] * view_weights[2]
            
            fused = np.concatenate([weighted_output, weighted_activation, weighted_input])
            fused_features.append(fused)
        
        fused_array = np.array(fused_features)
        
        # 一致性约束/对齐机制：融合后再整体标准化+可选PCA
        def post_fusion_alignment(fused_features, apply_pca=True, final_dim=None):
            """融合后一致性对齐：整体标准化 + 可选PCA"""
            from sklearn.preprocessing import StandardScaler
            
            # 整体标准化（确保不同视图融合后的数值一致性）
            scaler = StandardScaler()
            aligned_features = scaler.fit_transform(fused_features)
            alignment_info = "post_standardized"
            
            # 可选的整体PCA（进一步降维和去相关）
            if apply_pca and final_dim is not None:
                n_samples, total_dim = aligned_features.shape
                max_components = min(n_samples, total_dim)
                effective_final_dim = min(final_dim, max_components)
                
                if total_dim > effective_final_dim:
                    from sklearn.decomposition import PCA
                    final_pca = PCA(n_components=effective_final_dim, random_state=42)
                    aligned_features = final_pca.fit_transform(aligned_features)
                    final_explained = np.sum(final_pca.explained_variance_ratio_)
                    alignment_info += f"_pca{effective_final_dim}_var{final_explained:.3f}"
                
            return aligned_features, alignment_info
        
        # 应用融合后对齐（可选最终降维到150维，保持合理复杂度）
        final_features, alignment_method = post_fusion_alignment(fused_array, apply_pca=True, final_dim=150)
        
        # 直接K-means聚类（检验多模态聚类效果）
        kmeans = KMeans(n_clusters=2, random_state=0).fit(final_features)
        labels = kmeans.labels_
        
        # 使用LFD的聚类质量评估
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(final_features[i])
        
        # 计算聚类质量并生成权重
        cs0, cs1 = lfd.clusters_dissimilarity(clusters)
        good_cl = 0 if cs0 < cs1 else 1
        
        # 根据聚类质量选择好的cluster生成权重
        scores = []
        for i in range(m):
            if labels[i] == good_cl:  # 选择质量更好的cluster
                scores.append(1.0)
            else:  # 质量较差的cluster
                scores.append(0.0)  # 完全排除，与原始LFighter一致
        
        # 使用计算出的scores进行加权平均（移除循环导入）
        aggregated_weights = average_weights(local_weights, scores)
        
        # 返回聚合权重和视图权重信息
        view_weights_info = {
            'output_grad': float(view_weights[0]),
            'first_activation': float(view_weights[1]),
            'input_grad': float(view_weights[2])
        }
        
        return aggregated_weights, view_weights_info


class LFighterMVDBO:
    def __init__(self):
        pass
    
    def aggregate(self, simulation_model, local_weights, local_features):
        """LFighter-MV-DBO: 重用LFighterMV多视图特征提取 + DBONet训练 + LFD聚类质量评估"""
        import config
        device = simulation_model.device if hasattr(simulation_model, 'device') else config.DEVICE
        
        import torch
        from models import DBONet
        from sklearn.neighbors import kneighbors_graph
        
        # 检查输入
        if not local_features or len(local_features) == 0:
            raise ValueError("LFighter-MV-DBO requires local_features but got None or empty list.")
        
        m = len(local_weights)
        print(f"[LFighter-MV-DBO] Processing {m} clients with DBONet + LFD quality assessment")
        
        # === 步骤1: 重用LFighterMV的多视图特征构建逻辑 ===
        lfighter_mv = LFighterMV()
        
        # 创建一个临时的LFD实例用于聚类质量评估  
        lfd = LFD(config.NUM_CLASSES)
        
        # 调用LFighterMV来获取处理好的多视图特征（但不用它的聚类结果）
        # 我们需要手动调用LFighterMV的特征提取部分
        local_models = []
        for local_weight in local_weights:
            model = copy.deepcopy(simulation_model)
            model.load_state_dict(local_weight)
            local_models.append(list(model.parameters()))
        
        global_model = list(simulation_model.parameters())
        
        # 异常类别检测（与LFighterMV一致）
        memory = np.zeros(global_model[-1].shape[0])
        for i in range(m):
            dw = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            db = global_model[-1].cpu().data.numpy() - local_models[i][-1].cpu().data.numpy()
            norms = np.linalg.norm(dw, axis=-1)
            memory += norms + np.abs(db)
        
        max_two_freq_classes = memory.argsort()[-2:]
        print(f'[LFighter-MV-DBO] Detected anomalous classes: {max_two_freq_classes}')
        
        # 视图1: 输出层梯度
        output_grad_features = []
        for i in range(m):
            output_grad = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            key_classes_grad = output_grad[max_two_freq_classes].reshape(-1)
            output_grad_features.append(key_classes_grad)
        
        # 视图2: 第一层激活值
        first_activation_features = []
        for i, peer_features in enumerate(local_features):
            actual_features = peer_features
            if isinstance(peer_features, (list, tuple)) and len(peer_features) > 0:
                if isinstance(peer_features[0], (list, tuple)):
                    actual_features = peer_features[0]
            
            first_activation = actual_features[1]
            if not hasattr(first_activation, 'detach'):
                if isinstance(first_activation, (list, tuple)) and len(first_activation) > 0:
                    first_activation = first_activation[0]
            
            first_activation_flat = first_activation[0].detach().cpu().numpy().flatten()
            first_activation_features.append(first_activation_flat)
        
        # 视图3: 输入层梯度
        input_grad_features = []
        for i in range(m):
            input_grad = global_model[0].cpu().data.numpy() - local_models[i][0].cpu().data.numpy()
            input_grad_features.append(input_grad.reshape(-1))
        
        # === 步骤2: 统一降维处理 ===
        def unified_dimension_reduction(features, target_dim=200, standardize=True):
            features_array = np.array(features)
            n_samples, original_dim = features_array.shape
            
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_array = scaler.fit_transform(features_array)
            
            max_pca_components = min(n_samples, original_dim)
            effective_target_dim = min(target_dim, max_pca_components)
            
            if original_dim > effective_target_dim:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=effective_target_dim, random_state=42)
                features_array = pca.fit_transform(features_array)
                
            return features_array
        
        # 应用统一降维
        output_reduced = unified_dimension_reduction(output_grad_features, target_dim=200)
        activation_reduced = unified_dimension_reduction(first_activation_features, target_dim=200)
        input_reduced = unified_dimension_reduction(input_grad_features, target_dim=200)
        
        # === 步骤3: 视图权重计算（基于理论） ===
        view_weights = np.array([0.9, 0.05, 0.05])  # Output主导，与理论一致
        print(f"[LFighter-MV-DBO] View weights: Output={view_weights[0]:.3f}, Activation={view_weights[1]:.3f}, Input={view_weights[2]:.3f}")
        
        # === 步骤4: DBONet训练 ===
        # 应用权重并构建多视图特征
        view_features = [
            output_reduced * np.sqrt(view_weights[0]),
            activation_reduced * np.sqrt(view_weights[1]), 
            input_reduced * np.sqrt(view_weights[2])
        ]
        
        nfeats = [f.shape[1] for f in view_features]
        n_view = len(view_features)
        
        # 构建邻接矩阵
        adjs = []
        for v in range(n_view):
            n_neighbors = min(5, m-1)
            adj = kneighbors_graph(view_features[v], n_neighbors=n_neighbors, 
                                 mode='connectivity', include_self=True)
            adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
            adjs.append(adj_tensor)
        
        # DBONet配置
        n_clusters = 2
        blocks = 2
        para = 0.05
        np.random.seed(42)
        Z_init = np.random.randn(m, n_clusters) * 0.01
        
        # 创建和训练DBONet
        dbo_model = DBONet(nfeats, n_view, n_clusters, blocks, para, Z_init, device)
        features_tensor = [torch.tensor(view_features[v], dtype=torch.float32, device=device) 
                         for v in range(n_view)]
        
        # 归一化处理
        def normalization(data):
            maxVal = torch.max(data)
            minVal = torch.min(data)
            return (data - minVal) / (maxVal - minVal + 1e-10)

        def standardization(data):
            rowSum = torch.sqrt(torch.sum(data**2, 1))
            repMat = rowSum.repeat((data.shape[1], 1)) + 1e-10
            return torch.div(data, repMat.t())
        
        features_norm = []
        for i in range(n_view):
            feature = features_tensor[i]
            feature = (feature - feature.mean(dim=0)) / (feature.std(dim=0) + 1e-8)
            feature = standardization(normalization(feature))
            features_norm.append(feature)
        
        # 训练DBONet
        dbo_model.train()
        optimizer = torch.optim.Adam(dbo_model.parameters(), lr=5e-4, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(8):
            optimizer.zero_grad()
            output_z = dbo_model(features_tensor, adjs)
            
            # 加权损失计算
            loss_dis = torch.tensor(0., device=device)
            loss_lap = torch.tensor(0., device=device)
            
            for k in range(n_view):
                weight = float(view_weights[k])
                target_sim = features_norm[k] @ features_norm[k].t()
                pred_sim = output_z @ output_z.t()
                loss_dis += criterion(pred_sim, target_sim) * weight
                loss_lap += criterion(pred_sim, adjs[k]) * weight
            
            total_loss = loss_dis + 0.3 * loss_lap
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(dbo_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"[LFighter-MV-DBO] Epoch {epoch+1}: Loss={total_loss.item():.6f}")
        
        # === 步骤5: 使用LFD聚类质量评估替换简单kmeans ===
        dbo_model.eval()
        with torch.no_grad():
            output_z = dbo_model(features_tensor, adjs)
        
        z_np = output_z.detach().cpu().numpy()
        
        # 简化的kmeans聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(z_np)
        labels = kmeans.labels_
        
        # === 使用LFD的clusters_dissimilarity进行质量评估 ===
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(z_np[i])
        
        # 调用LFD的聚类质量评估方法
        cs0, cs1 = lfd.clusters_dissimilarity(clusters)
        good_cl = 0 if cs0 < cs1 else 1
        
        print(f"[LFighter-MV-DBO] LFD cluster quality: cs0={cs0:.4f}, cs1={cs1:.4f}, good_cluster={good_cl}")
        
        # 权重分配
        scores = np.ones(m)
        for i, l in enumerate(labels):
            if l != good_cl:
                scores[i] = 0
        
        print(f"[LFighter-MV-DBO] Good clients: {np.sum(scores)}/{m}")
        
        # 返回结果
        aggregated_weights = average_weights(local_weights, scores)
        view_weights_info = {
            'output_grad': float(view_weights[0]),
            'first_activation': float(view_weights[1]),
            'input_grad': float(view_weights[2])
        }
        
        return aggregated_weights, view_weights_info


class Tolpegin:
    def __init__(self):
        pass
    
    def score(self, global_model, local_models, peers_types, selected_peers):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)
        grads = [None for i in range(m)]
        for i in range(m):
            grad= (last_g - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy())
            grads[i] = grad
        
        grads = np.array(grads)
        num_classes = grad.shape[0]
        # print('Number of classes:', num_classes)
        dist = [ ]
        labels = [ ]
        for c in range(num_classes):
            data = grads[:, c]
            data = get_pca(copy.deepcopy(data))
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            cl = kmeans.cluster_centers_
            dist.append(((cl[0] - cl[1])**2).sum())
            labels.append(kmeans.labels_)
        
        dist = np.array(dist)
        candidate_class = dist.argmax()
        print("Candidate source/target class", candidate_class)
        labels = labels[candidate_class]
        if sum(labels) < m/2:
            scores = 1 - labels
        else:
            scores = labels
        
        for i, pt in enumerate(peers_types):
            print(pt, 'scored', scores[i])
        return scores
#################################################################################################################
# Clip local updates
def clipp_model(g_w, w, gamma =  1):
    for layer in w.keys():
        w[layer] = g_w[layer] + (w[layer] - g_w[layer])*min(1, gamma)
    return w
def FLAME(global_model, local_models, noise_scalar):
    # Compute number of local models
    m = len(local_models)
    
    # Flattent local models
    g_m = np.array([torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy()])
    f_m = np.array([torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy() for model in local_models])
    grads = g_m - f_m
    # Compute model-wise cosine similarity
    cs = smp.cosine_similarity(grads)
    # Compute the minimum cluster size value
    msc = int(m*0.5) + 1 
    # Apply HDBSCAN on the computed cosine similarities
    clusterer = hdbscan.HDBSCAN(min_cluster_size=msc, min_samples=1, allow_single_cluster = True)
    clusterer.fit(cs)
    labels = clusterer.labels_
    # print('Clusters:', labels)

    if sum(labels) == -(m):
        # In case all of the local models identified as outliers, consider all of as benign
        benign_idxs = np.arange(m)
    else:
        benign_idxs = np.where(labels!=-1)[0]
        
    # Compute euclidean distances to the current global model
    euc_d = cdist(g_m, f_m)[0]
    # Identify the median of computed distances
    st = np.median(euc_d)
    # Clipp admitted updates
    W_c = []
    for i, idx in enumerate(benign_idxs):
        w_c = clipp_model(global_model.state_dict(), local_models[idx].state_dict(), gamma =  st/euc_d[idx])
        W_c.append(w_c)
    
    # Average admitted clipped updates to obtain a new global model
    g_w = average_weights(W_c, np.ones(len(W_c)))
    
    '''From the original paper: {We use standard DP parameters and set eps = 3705 for IC, 
    eps = 395 for the NIDS and eps = 4191 for the NLP scenario. 
    Accordingly, lambda = 0.001 for IC and NLP, and lambda = 0.01 for the NIDS scenario.}
    However, we found lambda = 0.001 with the CIFAR10-ResNet18 benchmark spoils the model
    and therefore we tried lower lambda values, which correspond to greater eps values.'''
    
    # Add adaptive noise to the global model
    lamb = 0.001
    sigma = lamb*st*noise_scalar
    # print('Sigma:{:0.4f}'.format(sigma))
    for key in g_w.keys():
        noise = torch.FloatTensor(g_w[key].shape).normal_(mean=0, std=(sigma**2)).to(g_w[key].device)
        g_w[key] = g_w[key] + noise
        
    return g_w 
#################################################################################################################

def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med
        
# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med

def trimmed_mean(w, trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])
        
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg
   
def Krum(updates, f, multi = False):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
      updates_[i] = updates[i]
    k = n - f - 2
    # collection distance, distance from points to points
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k , largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
      return idxs[:k]
    else:
      return idxs[0]
##################################################################
