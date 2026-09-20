# -*- coding: utf-8 -*-
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import  HeteroConv, SAGEConv
import warnings
import sys, os
from dataset import get_numSMILES, get_numFASTA
from readfromKB import readfile
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/mol2vec')
warnings.filterwarnings('ignore')

#读取数据，为创建subgraph做准备
time_start = time.time()

# 读取KB数据集
drug_dict ,protein_dict ,DDI_dict,PPI_dict,DPI_dict = readfile()
initial_length_of_drug_dict = len(drug_dict)
initial_length_of_protein_dict = len(protein_dict)

# 对于每一个batch的数据，整理数据为input:[DrugID,ProteinID,numSMILES,numFASTA,label]
def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [],[]
    MAX_len_SMILES = 100
    MAX_len_FASTA = 1000
    numSMILESs = torch.zeros((N, MAX_len_SMILES), dtype=torch.long)
    numFASTAs = torch.zeros((N, MAX_len_FASTA), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i,pair in enumerate(batch_data):
        pair = pair.strip().split()
        drug_id,protein_id, smiles, fasta, label = pair[-5], pair[-4],pair[-3], pair[-2], pair[-1]

        drug_ids.append(drug_id)
        if drug_id not in drug_dict.keys():
            drug_dict.update({drug_id: smiles})

        protein_ids.append(protein_id)
        if protein_id not in protein_dict.keys():
            protein_dict.update({protein_id: fasta})

        compoundint = torch.from_numpy(get_numSMILES(smiles, MAX_len_SMILES))
        numSMILESs[i] = compoundint

        proteinint = torch.from_numpy(get_numFASTA(fasta, MAX_len_FASTA))
        numFASTAs[i] = proteinint

        label = float(label)
        labels_new[i] = int(label)

    return drug_ids, protein_ids,  numSMILESs, numFASTAs, labels_new

class SSGraphDTI(nn.Module):
    def __init__(self, hp):
        super(SSGraphDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv

        self.drug_kernel = hp.drug_kernel
        self.protein_kernel = hp.protein_kernel

        self.drug_MAX_LENGH = hp.drug_MAX_LENGH
        self.protein_MAX_LENGH = hp.protein_MAX_LENGH

        self.num_of_neighbourlayers_drug = hp.num_of_neighbourlayers_drug
        self.num_of_neighbours_drug = hp.num_of_neighbours_drug
        self.num_of_neighbourlayers_protein = hp.num_of_neighbourlayers_protein
        self.num_of_neighbours_protein = hp.num_of_neighbours_protein
        self.gnn_layers = hp.gnn_layers


        self.drug_embed = nn.Embedding(71, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)

        # 药物多尺度并行CNN：3个分支 kernel=3,5,7，各输出conv=40维，拼接后120维，映射到160维
        self.drug_branch1 = nn.Conv1d(self.dim, self.conv, kernel_size=3, padding=1)
        self.drug_branch2 = nn.Conv1d(self.dim, self.conv, kernel_size=5, padding=2)
        self.drug_branch3 = nn.Conv1d(self.dim, self.conv, kernel_size=7, padding=3)
        self.drug_proj = nn.Linear(self.conv * 3, 160)  # 120 -> 160

        # 靶点多尺度并行CNN：3个分支 kernel=4,8,12，各输出conv=40维，拼接后120维，映射到160维
        self.protein_branch1 = nn.Conv1d(self.dim, self.conv, kernel_size=4, padding=2)
        self.protein_branch2 = nn.Conv1d(self.dim, self.conv, kernel_size=8, padding=4)
        self.protein_branch3 = nn.Conv1d(self.dim, self.conv, kernel_size=12, padding=6)
        self.protein_proj = nn.Linear(self.conv * 3, 160)  # 120 -> 160

        #采用embedding进行编码
        self.creat_drug_node = nn.Embedding(num_embeddings=71,embedding_dim=self.drug_MAX_LENGH,padding_idx=0)
        self.creat_protein_node = nn.Embedding(num_embeddings=26, embedding_dim=self.protein_MAX_LENGH, padding_idx=0)

        self.graphConv = HeteroConv(
            {('drug', 'DDI', 'drug'):SAGEConv((-1, -1), self.drug_MAX_LENGH),
             ('protein', 'PPI', 'protein'):SAGEConv((-1, -1), self.protein_MAX_LENGH),
             ('drug', 'DPI', 'protein'):SAGEConv((-1, -1), self.protein_MAX_LENGH),
             ('protein', 'PDI', 'drug'):SAGEConv((-1, -1), self.drug_MAX_LENGH)}, aggr='mean')

        # ===== 第三阶段：加载离线 LLM 特征 =====
        # 将特征字典直接常驻 GPU 显存，彻底消除前向传播时的 PCIe 拷贝瓶颈
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        raw_drug_feats = torch.load('data/drug_molformer_features.pt')
        self.drug_llm_feats = {k: v.to(device) for k, v in raw_drug_feats.items()}

        raw_protein_feats = torch.load('data/protein_prott5_features.pt')
        self.protein_llm_feats = {k: v.to(device) for k, v in raw_protein_feats.items()}

        # 均值也常驻 GPU
        drug_feats_list = list(self.drug_llm_feats.values())
        self.drug_llm_mean = torch.stack(drug_feats_list).mean(dim=0)  # (768,)
        protein_feats_list = list(self.protein_llm_feats.values())
        self.protein_llm_mean = torch.stack(protein_feats_list).mean(dim=0)  # (1024,)

        # 药物 LLM 特征投影层：768 -> 160
        self.drug_llm_proj = nn.Sequential(nn.Linear(768, 160), nn.ReLU())
        # 靶点 LLM 特征投影层：1024 -> 160
        self.protein_llm_proj = nn.Sequential(nn.Linear(1024, 160), nn.ReLU())

        # 药物门控层：拼接 CNN(160) + LLM(160) = 320 -> 输出 160 维门控权重
        self.drug_gate = nn.Sequential(nn.Linear(160 + 160, 160), nn.Sigmoid())
        # 靶点门控层：拼接 CNN(160) + LLM(160) = 320 -> 输出 160 维门控权重
        self.protein_gate = nn.Sequential(nn.Linear(160 + 160, 160), nn.Sigmoid())

        self.dropout1 = nn.Dropout(hp.FC_Dropout)
        self.dropout2 = nn.Dropout(hp.FC_Dropout)
        self.dropout3 = nn.Dropout(hp.FC_Dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.bn_pair = nn.BatchNorm1d(1420)
        self.fc1 = nn.Linear(1420, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    # 创建与输入相关的子网络；输入：药物ID，药物邻居跳数，层药物邻居最大数，蛋白ID，蛋白邻居跳数，层蛋白邻居最大数
    # 返回：ddi_edgeindex, ppi_edgeindex, dpi_edgeindex, pdi_edgeindex, temp_druglist, temp_prolist
    def creat_subgraph(self, drugID, num_NeighbourLayers_of_drug, num_Neighbours_of_drug,
                       proID, num_NeighbourLayers_of_protein,num_Neighbours_of_protein):
        # find temp_druglist
        temp_druglist = []
        temp_druglist.append(drugID)
        if drugID in DDI_dict.keys():
            for i in range(num_NeighbourLayers_of_drug):
                addlist_drug = []
                for item in temp_druglist:
                    if item in DDI_dict.keys():
                        if len(DDI_dict[item])>= num_Neighbours_of_drug:
                            add_drugs_indexs = random.sample(range(0,len(DDI_dict[item])),num_Neighbours_of_drug)
                            for index_drug in add_drugs_indexs:
                                addlist_drug.append(DDI_dict[item][index_drug])
                        else:
                            addlist_drug.extend(DDI_dict[item])
                addlist_drug = set(addlist_drug).difference(temp_druglist)
                temp_druglist.extend(addlist_drug)

        # find temp_prolist
        temp_prolist = []
        temp_prolist.append(proID)
        if proID in PPI_dict.keys():
            for i in range(num_NeighbourLayers_of_protein):
                addlist_pro = []
                for item in temp_prolist:
                    if item in PPI_dict.keys():
                        if len(PPI_dict[item])>=num_Neighbours_of_protein:
                            add_proteins_indexs = random.sample(range(0, len(PPI_dict[item])), num_Neighbours_of_protein)
                            for index_protein in add_proteins_indexs:
                                addlist_pro.append(PPI_dict[item][index_protein])
                        else:
                            addlist_pro.extend(PPI_dict[item])
                addlist_pro = set(addlist_pro).difference(temp_prolist)
                temp_prolist.extend(addlist_pro)

        # create DDI index
        ddi_edgeindex = None
        if len(temp_druglist) > 1:
            DDI_coo = []
            for drug1 in temp_druglist:
                if drug1 in DDI_dict.keys():
                    drug2list = DDI_dict[drug1]
                    for item in drug2list:
                        if item in temp_druglist:
                            newpoint = [[temp_druglist.index(drug1), temp_druglist.index(item)]]
                            DDI_coo.extend(newpoint)
            ddi_edgeindex = torch.tensor(DDI_coo).t()

        # create PPI index
        ppi_edgeindex = None
        if len(temp_prolist) > 1:
            PPI_coo = []
            for pro1 in temp_prolist:
                if pro1 in PPI_dict.keys():
                    pro2list = PPI_dict[pro1]
                    for item in pro2list:
                        if item in temp_prolist:
                            newpoint = [[temp_prolist.index(pro1), temp_prolist.index(item)]]
                            PPI_coo.extend(newpoint)
            ppi_edgeindex = torch.tensor(PPI_coo).t()

        # create DPI index
        DPI_coo = []
        for drug1 in temp_druglist:
            if drug1 in DPI_dict.keys():
                pro2list = DPI_dict[drug1]
                for item in pro2list:
                    if item in temp_prolist:
                        newpoint = [[temp_druglist.index(drug1), temp_prolist.index(item)]]
                        DPI_coo.extend(newpoint)
        if len(DPI_coo) == 0:
            dpi_edgeindex = None
        else:
            dpi_edgeindex = torch.tensor(DPI_coo).t()
        # create PDI index
        if dpi_edgeindex == None:
            pdi_edgeindex = None
        else:
            pdi_edgeindex = dpi_edgeindex[[1, 0]]

        return ddi_edgeindex, ppi_edgeindex, dpi_edgeindex, pdi_edgeindex, temp_druglist, temp_prolist

    #
    def DP_Pair_graphConv(self, drugIDs, proteinIDs, graph_model_layers_num):
        result = torch.zeros((len(drugIDs), self.protein_MAX_LENGH+self.drug_MAX_LENGH))
        for i in range(len(drugIDs)):
            ddi_edgeindex, ppi_edgeindex, dpi_edgeindex, pdi_edgeindex, temp_druglist, temp_prolist = \
                self.creat_subgraph(drugIDs[i],self.num_of_neighbourlayers_drug,self.num_of_neighbours_drug,
            proteinIDs[i],self.num_of_neighbourlayers_protein,self.num_of_neighbours_protein)
            data = HeteroData()
            data['drug', 'DDI', 'drug'].edge_index = ddi_edgeindex
            data['protein', 'PPI', 'protein'].edge_index = ppi_edgeindex
            data['drug', 'DPI', 'protein'].edge_index = dpi_edgeindex
            data['protein', 'PDI', 'drug'].edge_index = pdi_edgeindex
            data['drug'].x = torch.zeros((len(temp_druglist),self.drug_MAX_LENGH ))
            data['protein'].x = torch.zeros((len(temp_prolist),self.protein_MAX_LENGH))

            for j, drugname in enumerate(temp_druglist):
                data['drug'].x[j] = torch.from_numpy((get_numSMILES(drug_dict[drugname], self.drug_MAX_LENGH)))
            data['drug'].x = data['drug'].x.int().cuda()
            data['drug'].x = self.creat_drug_node(data['drug'].x)
            data['drug'].x = torch.mean(data['drug'].x, dim=1)

            for k, proname in enumerate(temp_prolist):
                data['protein'].x[k] =  torch.from_numpy((get_numFASTA(protein_dict[proname],self.protein_MAX_LENGH)))
            data['protein'].x = data['protein'].x.int().cuda()
            data['protein'].x = self.creat_protein_node( data['protein'].x)
            data['protein'].x = torch.mean( data['protein'].x,dim=1)

            data = data.cuda()
            graphconv_out = self.graphConv(data.collect('x'),data.collect("edge_index"))

            if graph_model_layers_num > 1:
                for conv_num in range(graph_model_layers_num - 1):
                    if len(graphconv_out['drug']) == 0 and len(graphconv_out['protein']) == 0:# drug、protein均不更新
                        result[i] = torch.cat(( data['drug'].x[0],  data['protein'].x[0]), dim=0)

                    if len(graphconv_out['drug']) > 0 and len(graphconv_out['protein']) == 0:# drug 更新
                        data['drug'].x = graphconv_out['drug']
                        graphconv_out = self.graphConv(data.collect('x'), data.collect("edge_index"))
                        result[i] = torch.cat((graphconv_out['drug'][0], data['protein'].x[0]), dim=0)

                    if len(graphconv_out['drug']) == 0 and len(graphconv_out['protein']) > 0:# protein 更新
                        data['protein'].x = graphconv_out['protein']
                        graphconv_out = self.graphConv(data.collect('x'), data.collect("edge_index"))
                        result[i] = torch.cat((data['drug'].x[0], graphconv_out['protein'][0]), dim=0)

                    if  len(graphconv_out['drug']) > 0 and len(graphconv_out['protein']) > 0:# drug、protein均更新
                        data['drug'].x = graphconv_out['drug']
                        data['protein'].x = graphconv_out['protein']
                        graphconv_out = self.graphConv(data.collect('x'), data.collect("edge_index"))
                        result[i] = torch.cat((graphconv_out['drug'][0], graphconv_out['protein'][0]), dim=0)

            if graph_model_layers_num ==1 :
                if len(graphconv_out['drug']) == 0  and len(graphconv_out['protein']) == 0:
                    result[i] = torch.cat(( data['drug'].x[0],  data['protein'].x[0]), dim=0)
                if len(graphconv_out['drug']) > 0  and len(graphconv_out['protein']) == 0:
                    result[i] = torch.cat((graphconv_out['drug'][0],  data['protein'].x[0]), dim=0)
                if len(graphconv_out['drug']) == 0 and len(graphconv_out['protein']) > 0:
                    result[i] = torch.cat((data['drug'].x[0], graphconv_out['protein'][0]), dim=0)
                if  len(graphconv_out['drug']) > 0 and len(graphconv_out['protein']) > 0:
                    result[i] = torch.cat((graphconv_out['drug'][0], graphconv_out['protein'][0]), dim=0)
        return result

    def forward(self, drugIDs, proIDs, numSMILES,numFASTA):
        drugembed = self.drug_embed(numSMILES)
        proteinembed = self.protein_embed(numFASTA)

        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # 药物多尺度特征提取：3分支并行卷积 -> 全局最大池化 -> 拼接 -> 线性映射到160维
        d1 = torch.relu(self.drug_branch1(drugembed)).max(dim=2)[0]
        d2 = torch.relu(self.drug_branch2(drugembed)).max(dim=2)[0]
        d3 = torch.relu(self.drug_branch3(drugembed)).max(dim=2)[0]
        drugConv = self.relu(self.drug_proj(self.dropout1(torch.cat([d1, d2, d3], dim=1))))

        # 靶点多尺度特征提取：3分支并行卷积 -> 全局最大池化 -> 拼接 -> 线性映射到160维
        p1 = torch.relu(self.protein_branch1(proteinembed)).max(dim=2)[0]
        p2 = torch.relu(self.protein_branch2(proteinembed)).max(dim=2)[0]
        p3 = torch.relu(self.protein_branch3(proteinembed)).max(dim=2)[0]
        proteinConv = self.relu(self.protein_proj(self.dropout2(torch.cat([p1, p2, p3], dim=1))))

        graphConv = self.DP_Pair_graphConv(drugIDs, proIDs,self.gnn_layers)
        graphConv = graphConv.cuda().float()

        # ===== 第三阶段：自适应特征门控融合 =====
        # --- 药物 LLM 特征提取 ---
        # 根据 drugIDs 从字典中取出对应的 LLM 特征，查不到则用均值填充
        drug_llm_list = [
            self.drug_llm_feats.get(did, self.drug_llm_mean) for did in drugIDs
        ]
        drug_llm_batch = torch.stack(drug_llm_list)  # (B, 384)
        drug_llm_emb = self.drug_llm_proj(drug_llm_batch)                # (B, 160)

        # --- 靶点 LLM 特征提取 ---
        protein_llm_list = [
            self.protein_llm_feats.get(pid, self.protein_llm_mean) for pid in proIDs
        ]
        protein_llm_batch = torch.stack(protein_llm_list)  # (B, 320)
        protein_llm_emb = self.protein_llm_proj(protein_llm_batch)                # (B, 160)

        # --- 核心门控融合逻辑 ---
        # 将 CNN 局部特征与 LLM 全局特征拼接，计算自适应门控权重 G
        drug_G = self.drug_gate(torch.cat([drugConv, drug_llm_emb], dim=1))          # (B, 160)
        final_drug_feat = drug_G * drugConv + (1 - drug_G) * drug_llm_emb            # (B, 160)

        protein_G = self.protein_gate(torch.cat([proteinConv, protein_llm_emb], dim=1))  # (B, 160)
        final_protein_feat = protein_G * proteinConv + (1 - protein_G) * protein_llm_emb  # (B, 160)

        # 融合后特征与图卷积特征拼接：160 + 160 + 1100 = 1420，与 FC 层完美对接
        pair = torch.cat([final_drug_feat, final_protein_feat, graphConv], dim=1)

        pair = self.bn_pair(pair)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))

        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))

        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))

        predict = self.out(fully3)
        return predict
