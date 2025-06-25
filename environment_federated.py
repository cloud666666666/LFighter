from __future__ import print_function
from lib2to3.pgen2.tokenize import tokenize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from models import *
from utils import *
from sampling import *
from datasets import *
import os
import random
from tqdm import tqdm
import copy
from operator import itemgetter
import time
from random import shuffle
from aggregation import *
from IPython.display import clear_output
import gc
import logging
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
# from datetime import datetime  # 已移除时间戳，不再需要

# DBONet 数据归一化函数 (来自原文)
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

class Peer():
    # Class variable shared among all the instances
    _performed_attacks = 0
    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self,val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, local_data, labels, criterion, 
                device, local_epochs, local_bs, local_lr, 
                local_momentum, peer_type = 'honest'):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.peer_type = peer_type
#======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type = 'no_attack', malicious_behavior_rate = 0, 
                            source_class = None, target_class = None, dataset_name = None, rule = 'fedavg') :
        epochs = self.local_epochs
        train_loader = DataLoader(self.local_data, self.local_bs, shuffle = True, drop_last=True)
        attacked = 0
        local_features = []  # 新增：收集多视图特征
        #Get the poisoned training data of the peer in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.peer_type == 'attacker'):
            r = np.random.random()
            if r <= malicious_behavior_rate:
                if dataset_name != 'IMDB':
                    poisoned_data = label_filp(self.local_data, source_class, target_class)
                    train_loader = DataLoader(poisoned_data, self.local_bs, shuffle = True, drop_last=True)
                self.performed_attacks+=1
                attacked = 1
                print('Label flipping attack launched by', self.peer_pseudonym, 'to flip class ', source_class,
                ' to class ', target_class)
        lr=self.local_lr

        if dataset_name == 'IMDB':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=self.local_momentum, weight_decay=5e-4)
        model.train()
        epoch_loss = []
        peer_grad = []
        t = 0
        for epoch in range(epochs):
            correct, total = 0, 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if dataset_name == 'IMDB':
                    target = target.view(-1,1) * (1 - attacked)

                data, target = data.to(self.device), target.to(self.device)
                if target.ndim > 1:
                    target = target.argmax(dim=-1)

                # 只有在rule包含'mv'时才提取多视图特征
                if 'mv' in rule and hasattr(model, 'forward') and 'return_features' in model.forward.__code__.co_varnames:
                    features, output = model(data, return_features=True)
                    # 只保存最后一个batch的特征作为代表（避免过多的特征累积）
                    if batch_idx == len(train_loader) - 1:  # 最后一个batch
                        local_features.append(features)
                else:
                    output = model(data)
                loss = self.criterion(output, target)
                loss.backward()    
                epoch_loss.append(loss.item())
                # 统计本地acc
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                # get gradients
                cur_time = time.time()
                for i, (name, params) in enumerate(model.named_parameters()):
                    if params.requires_grad:
                        if epoch == 0 and batch_idx == 0:
                            peer_grad.append(params.grad.clone())
                        else:
                            peer_grad[i]+= params.grad.clone()   
                t+= time.time() - cur_time    
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
            # 每个epoch结束后打印本地loss/acc
            avg_loss = np.mean(epoch_loss[-len(train_loader):])
            acc = correct / total if total > 0 else 0
            # print(f'[Peer {self.peer_id}] Local Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc*100:.2f}%')
    
        if (attack_type == 'gaussian' and self.peer_type == 'attacker'):
            update, flag =  gaussian_attack(model.state_dict(), self.peer_pseudonym,
            malicious_behavior_rate = malicious_behavior_rate, device = self.device)
            if flag == 1:
                self.performed_attacks+=1
                attacked = 1
            model.load_state_dict(update)

        model = model.cpu()
        return model.state_dict(), peer_grad , model, np.mean(epoch_loss), attacked, t, local_features
#======================================= End of training function =============================================================#
#========================================= End of Peer class ====================================================================


class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_peers, frac_peers, 
    seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, labels_dict, device, attackers_ratio = 0,
    class_per_peer=2, samples_per_class= 250, rate_unbalance = 1, alpha = 1,source_class = None):

        FL._history = np.zeros(num_peers)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.peers_pseudonyms = ['Peer ' + str(i+1) for i in range(self.num_peers)]
        self.frac_peers = frac_peers
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_peer = class_per_peer
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.peers = []
        self.trainset, self.testset = None, None
        
        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       
        #Loading of data
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name, self.num_peers, self.num_classes, 
        self.dd_type, self.class_per_peer, self.samples_per_class, self.alpha)

        # 自动检查模型和数据尺寸是否匹配
        tmp_loader = DataLoader(self.trainset, batch_size=4, shuffle=True)
        inputs, _ = next(iter(tmp_loader))

        self.test_loader = DataLoader(self.testset, batch_size = self.test_batch_size,
            shuffle = False, num_workers = 1)
    
        #Creating model
        self.global_model = setup_model(model_architecture = self.model_name, num_classes = self.num_classes, 
        tokenizer = tokenizer, embedding_dim = self.embedding_dim)
        self.global_model = self.global_model.to(self.device)
        
        # Dividing the training set among peers
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        print('--> Distributing training data among peers')
        
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            peer_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(peer_data)
            if  self.source_class in user_groups_train[p]['labels']:
                 self.have_source_class.append(p)
        print('--> Training data have been distributed among peers')

        # Creating peers instances
        print('--> Creating peets instances')
        m_ = 0
        if self.attackers_ratio > 0:
            #pick m random participants from the workers list
            # k_src = len(self.have_source_class)
            # print('# of peers who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * self.num_peers)
            self.num_attackers = copy.deepcopy(m_)

        peers = list(np.arange(self.num_peers))  
        random.shuffle(peers)
        for i in peers:
            if m_ > 0 and contains_class(self.local_data[i], self.source_class):
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, peer_type = 'attacker'))
                m_-= 1
            else:
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum))  

        del self.local_data

    #======================================= Multi-view and DBO Aggregation Functions =========================================#



    
    def _parameter_similarity_aggregation(self, simulation_model, local_weights):
        """基于参数相似性的聚合方法 - 无回退机制"""
        param_similarities = []
        global_params = list(simulation_model.parameters())[0].detach().cpu().numpy().flatten()
        
        for i, local_weight in enumerate(local_weights):
            local_param = list(local_weight.values())[0].cpu().numpy().flatten()
            similarity = np.dot(global_params, local_param) / (np.linalg.norm(global_params) * np.linalg.norm(local_param) + 1e-8)
            param_similarities.append(abs(similarity))
        
        weights = np.array(param_similarities)
        weights = weights / (np.sum(weights) + 1e-8)
        
        print(f"[Parameter Similarity] Using weights: {weights}")
        return average_weights(local_weights, weights)



#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name=None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)

            if dataset_name == 'IMDB':
                # IMDB二分类/回归
                target_loss = target.float().view(-1, 1)
                loss = self.criterion(output, target_loss)
                pred = (output > 0.5).long().view(-1)
                target_acc = target.view(-1).long()
            else:
                # 多分类，统一target为类别索引
                if target.ndim > 1:
                    target = target.squeeze()  # 修复：使用squeeze而不是argmax
                loss = self.criterion(output, target)
                pred = output.argmax(dim=1)
                target_acc = target

            test_loss.append(loss.item())
            correct += (pred == target_acc).sum().item()
            n += target_acc.size(0)

        test_loss = np.mean(test_loss) if test_loss else float('inf')
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, n, 100 * correct / n))
        return 100.0 * (correct / n), test_loss
    #======================================= End of testning function =============================================================#
#Test label prediction function    
    def test_label_predictions(self, model, device, test_loader, dataset_name = None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if dataset_name == 'IMDB':
                    prediction = output > 0.5
                    target_processed = target.float().view(-1)
                else:
                    prediction = output.argmax(dim=1)
                    # 确保target是正确的类别索引格式
                    if target.ndim > 1:
                        target_processed = target.squeeze()  # 压缩维度而不是argmax
                    else:
                        target_processed = target
                
                # 确保转换为列表时保持正确的数据格式
                actuals.extend(target_processed.cpu().numpy().tolist())
                predictions.extend(prediction.cpu().numpy().tolist())
        return actuals, predictions
    
    #choose random set of peers
    def choose_peers(self):
        #pick m random peers from the available list of peers
        m = max(int(self.frac_peers * self.num_peers), 1)
        selected_peers = np.random.choice(range(self.num_peers), m, replace=False)

        # print('\nSelected Peers\n')
        # for i, p in enumerate(selected_peers):
        #     print(i+1, ': ', self.peers[p].peer_pseudonym, ' is ', self.peers[p].peer_type)
        return selected_peers

        
    def run_experiment(self, attack_type = 'no_attack', malicious_behavior_rate = 0,
        source_class = None, target_class = None, rule = 'fedavg', resume = False, log_file="experiment.log"):
        # 时间戳已移除，使用固定命名方式
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logging.basicConfig(filename=log_file,
                            filemode='a',
                            format='%(asctime)s %(message)s',
                            level=logging.INFO)
        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation started...')
        lfd = LFD(self.num_classes)
        fg = FoolsGold(self.num_peers)
        tolpegin = Tolpegin()
        lfighter_dbo = LFighterDBO()
        lfighter_mv = LFighterMV()
        lfighter_mv_dbo = LFighterMVDBO()
        lfighter_ae = LFighterAutoencoder(self.num_classes)
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        noise_scalar = 1.0
        asr = 0.0  # 初始化ASR变量
        # best_accuracy = 0.0
        mapping = {'honest': 'Good update', 'attacker': 'Bad update'}

        #start training
        start_round = 0
        if resume:
            print('Loading last saved checkpoint..')
            checkpoint = torch.load('./checkpoints/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']
            
            print('>>checkpoint loaded!')
        print("\n====>Global model training started...\n")
        for epoch in tqdm(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()
            
            # 初始化视图权重字符串
            view_weights_str = None
            
            # if epoch % 20 == 0:
            #     clear_output()  
            print(f'\n | Global training round : {epoch+1}/{self.global_rounds} |\n')
            selected_peers = self.choose_peers()
            local_weights, local_grads, local_models, local_losses, performed_attacks = [], [], [], [], []  
            all_local_features = []  # 新增：收集所有peer的特征
            peers_types = []
            i = 1        
            attacks = 0
            Peer._performed_attacks = 0
            for peer in selected_peers:
                peers_types.append(mapping[self.peers[peer].peer_type])
                peer_update, peer_grad, peer_local_model, peer_loss, attacked, t, peer_features = self.peers[peer].participant_update(epoch, 
                copy.deepcopy(simulation_model),
                attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                source_class = source_class, target_class = target_class, dataset_name = self.dataset_name, rule = rule)
                local_weights.append(peer_update)
                local_grads.append(peer_grad)
                local_losses.append(peer_loss) 
                local_models.append(peer_local_model)
                all_local_features.append(peer_features)  # 收集每个peer的特征
                attacks+= attacked
                # print('{} ends training in global round:{} |\n'.format((self.peers_pseudonyms[peer]), (epoch + 1))) 
                i+= 1
            # loss_avg = sum(local_losses) / len(local_losses)
            # print('Average of peers\' local losses: {:.6f}'.format(loss_avg))
            #aggregated global weights
            scores = np.zeros(len(local_weights))
            # Expected malicious peers
            f = int(self.num_peers*self.attackers_ratio)
            if rule == 'median':
                    cur_time = time.time()
                    global_weights = simple_median(local_weights)
                    cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'rmedian':
                cur_time = time.time()
                global_weights = Repeated_Median_Shard(local_weights)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'tmean':
                    cur_time = time.time()
                    trim_ratio = self.attackers_ratio*self.num_peers/len(selected_peers)
                    global_weights = trimmed_mean(local_weights, trim_ratio = trim_ratio)
                    cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'mkrum':
                cur_time = time.time()
                goog_updates = Krum(local_models, f = f, multi=True)
                scores[goog_updates] = 1
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'foolsgold':
                cur_time = time.time()
                scores = fg.score_gradients(local_grads, selected_peers)
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time + t)

            elif rule == 'Tolpegin':
                cur_time = time.time()
                scores = tolpegin.score(copy.deepcopy(self.global_model), 
                                            copy.deepcopy(local_models), 
                                            peers_types = peers_types,
                                            selected_peers = selected_peers)
                global_weights = average_weights(local_weights, scores)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)
            
            elif rule == 'FLAME':
                cur_time = time.time()
                global_weights = FLAME(copy.deepcopy(self.global_model).cpu(), copy.deepcopy(local_models), noise_scalar)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)


            elif rule == 'lfighter':
                cur_time = time.time()
                global_weights = lfd.aggregate(copy.deepcopy(simulation_model), copy.deepcopy(local_models), peers_types)
                cpu_runtimes.append(time.time() - cur_time)


            elif rule == 'lfighter_mv':
                cur_time = time.time()
                result = lfighter_mv.aggregate(simulation_model, local_weights, all_local_features, local_models, peers_types, lfd)
                if isinstance(result, tuple):
                    global_weights, view_weights_info = result
                    # 格式化视图权重信息用于日志
                    view_weights_str = f"Output:{view_weights_info['output_grad']:.3f},Activation:{view_weights_info['first_activation']:.3f},Input:{view_weights_info['input_grad']:.3f}"
                else:
                    global_weights = result
                    view_weights_str = "N/A"
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'lfighter_dbo':
                cur_time = time.time()
                global_weights = lfighter_dbo.aggregate(simulation_model, local_weights, all_local_features)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'lfighter_mv_dbo':
                cur_time = time.time()
                result = lfighter_mv_dbo.aggregate(simulation_model, local_weights, all_local_features)
                if isinstance(result, tuple):
                    global_weights, view_weights_info = result
                    # 格式化视图权重信息用于日志
                    view_weights_str = f"Output:{view_weights_info['output_grad']:.3f},Activation:{view_weights_info['first_activation']:.3f},Input:{view_weights_info['input_grad']:.3f}"
                else:
                    global_weights = result
                    view_weights_str = "N/A"
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'lfighter_ae':
                cur_time = time.time()
                global_weights = lfighter_ae.aggregate(copy.deepcopy(simulation_model), copy.deepcopy(local_models), peers_types)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'fedavg':
                cur_time = time.time()
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                cpu_runtimes.append(time.time() - cur_time)
            
            else:
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                ##############################################################################################
            #Plot honest vs attackers
            # if attack_type == 'label_flipping' and epoch >= 10 and epoch < 20:
            #     plot_updates_components(local_models, peers_types, epoch=epoch+1)   
            #     plot_layer_components(local_models, peers_types, epoch=epoch+1, layer = 'linear_weight')  
            #     plot_source_target(local_models, peers_types, epoch=epoch+1, classes= [source_class, target_class])
            # update global weights
            g_model = copy.deepcopy(simulation_model)
            simulation_model.load_state_dict(global_weights)           
            # 聚合后打印全局模型参数均值/方差
            params = list(simulation_model.parameters())
            print(f'[After aggregation] Epoch {epoch+1} global param mean: {params[0].data.mean().item():.6f}, std: {params[0].data.std().item():.6f}')
            if epoch >= self.global_rounds-10:
                last10_updates.append(global_weights) 

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            
            # 移除NaN检查回退机制 - 让真实错误显示
            
            # 计算当前轮次的ASR (在日志记录之前)
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(actuals, predictions, labels=classes)
            
            current_asr = 0.0  # 当前轮次的ASR
            current_source_acc = 0.0  # 当前轮次的源类别准确率
            for i, r in enumerate(cm):
                if i == source_class:
                    current_source_acc = np.round(r[i]/np.sum(r)*100, 2) if np.sum(r) > 0 else 0.0
                    source_class_accuracies.append(current_source_acc)
                    # 计算ASR
                    if np.sum(r) > 0:
                        current_asr = np.round(r[target_class]/np.sum(r)*100, 2)
                    break
            
            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
            performed_attacks.append(attacks) 
            
            # 每轮训练结束后写日志 (现在包含ASR)
            if (rule == 'lfighter_mv' or rule == 'lfighter_mv_dbo') and 'view_weights_str' in locals() and view_weights_str is not None:
                log_msg = (
                    f"Round {epoch+1}/{self.global_rounds} | "
                    f"Global Acc: {current_accuracy:.2f} | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Attacks: {attacks} | "
                    f"Source Class Acc: {current_source_acc:.2f} | "
                    f"ASR: {current_asr:.2f}% | "
                    f"ViewWeights: {view_weights_str}"
                )
            else:
                log_msg = (
                    f"Round {epoch+1}/{self.global_rounds} | "
                    f"Global Acc: {current_accuracy:.2f} | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Attacks: {attacks} | "
                    f"Source Class Acc: {current_source_acc:.2f} | "
                    f"ASR: {current_asr:.2f}%"
                )
            print(log_msg)
            logging.info(log_msg)
            
            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model':g_model,
                'local_models':copy.deepcopy(local_models),
                'last10_updates':last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies
                }
            savepath = './checkpoints/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
            torch.save(state,savepath)

            del local_models
            del local_weights
            del local_grads
            gc.collect()
            torch.cuda.empty_cache()
            # print("***********************************************************************************")
            
            # 详细混淆矩阵分析 (可选显示)
            # print("\n=== 完整混淆矩阵分析 ===")
            # print("混淆矩阵 (行=真实标签, 列=预测标签):")
            # print(cm)
            # print("\n各类别样本数量分布:")
            total_samples = 0
            for i, class_name in enumerate(classes):
                class_total = cm[i].sum()
                total_samples += class_total
            #     print(f"{self.labels_dict[class_name]:25}: {class_total:4d} 样本")
            # print(f"{'总样本数':25}: {total_samples:4d}")
            
            print('\n{0:10s} - {1:>8s} - {2:>8s} - {3:>8s}'.format('Class','Accuracy','Correct','Total'))
            print('-' * 60)
            correct_predictions = 0
            for i, r in enumerate(cm):
                class_accuracy = r[i]/np.sum(r)*100 if np.sum(r) > 0 else 0
                correct_predictions += r[i]
                print('{:25} - {:6.1f}% - {:6d} - {:6d}'.format(
                    self.labels_dict[classes[i]], 
                    class_accuracy,
                    r[i], 
                    np.sum(r)
                ))
                if i == source_class:
                    print(f"[轮次{epoch+1} ASR] 源类别{source_class}->目标类别{target_class}: {current_asr:.2f}%")
            
            # 验证全局准确率计算
            manual_global_acc = correct_predictions / total_samples * 100
            # print(f"\n手动计算的全局准确率: {manual_global_acc:.2f}%")
            print(f"test函数计算的全局准确率: {current_accuracy:.2f}%")
            # print("=== 混淆矩阵分析结束 ===\n")

            if epoch == self.global_rounds-1:
                print('Last 10 updates results')
                global_weights = average_weights(last10_updates, 
                np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights) 
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                performed_attacks.append(attacks)
                print("***********************************************************************************")
                #print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(actuals, predictions, labels=classes)
                
                # 最终轮次的完整混淆矩阵输出
                # print("\n=== 最终轮次完整混淆矩阵分析 ===")
                # print("混淆矩阵 (行=真实标签, 列=预测标签):")
                # print(cm)
                # print("\n各类别样本数量分布:")
                total_samples = 0
                for i, class_name in enumerate(classes):
                    class_total = cm[i].sum()
                    total_samples += class_total
                    # print(f"{self.labels_dict[class_name]:25}: {class_total:4d} 样本")
                # print(f"{'总样本数':25}: {total_samples:4d}")
                
                print('\n{0:10s} - {1:>8s} - {2:>8s} - {3:>8s}'.format('Class','Accuracy','Correct','Total'))
                print('-' * 60)
                correct_predictions = 0
                asr = 0.0  # 初始化ASR
                for i, r in enumerate(cm):
                    class_accuracy = r[i]/np.sum(r)*100 if np.sum(r) > 0 else 0
                    correct_predictions += r[i]
                    print('{:25} - {:6.1f}% - {:6d} - {:6d}'.format(
                        self.labels_dict[classes[i]], 
                        class_accuracy,
                        r[i], 
                        np.sum(r)
                    ))
                    if i == source_class:
                        source_class_accuracies.append(np.round(class_accuracy, 2))
                        # 修正ASR计算：源类别样本中被预测为目标类别的比例
                        if np.sum(r) > 0:
                            asr = np.round(r[target_class]/np.sum(r)*100, 2)
                        else:
                            asr = 0.0
                        print(f"[ASR计算] 源类别{source_class}总样本: {np.sum(r)}, 被预测为目标类别{target_class}: {r[target_class]}, ASR: {asr:.2f}%")
                
                # 验证全局准确率计算
                manual_global_acc = correct_predictions / total_samples * 100
                # print(f"\n手动计算的全局准确率: {manual_global_acc:.2f}%")
                print(f"test函数计算的全局准确率: {current_accuracy:.2f}%")
                # print("=== 最终混淆矩阵分析结束 ===\n")

        # 实验结束后写最终结果
        final_msg = (
            f"Experiment End | Rule: {rule} | Global Accuracies: {global_accuracies} | "
            f"Source Class Accuracies: {source_class_accuracies} | Test Losses: {test_losses}"
        )
        print(final_msg)
        logging.info(final_msg)

        state = {
                'state_dict': simulation_model.state_dict(),
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'asr':asr,
                'avg_cpu_runtime':np.mean(cpu_runtimes)
                }
        savepath = './results/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
        torch.save(state,savepath)            
        print('Global accuracies: ', global_accuracies)
        print('Class {} accuracies: '.format(source_class), source_class_accuracies)
        print('Test loss:', test_losses)
        print('Attack succes rate:', asr)
        print('Average CPU aggregation runtime:', np.mean(cpu_runtimes))
        
        # 返回最终结果给调用者
        final_accuracy = global_accuracies[-1] if global_accuracies else 0.0
        final_asr = asr if 'asr' in locals() else 0.0
        return final_accuracy, final_asr
