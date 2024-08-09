# _*_ coding:utf-8 _*_
import numpy as np
from tqdm import tqdm
from scipy import optimize
import tensorflow as tf
from utils import *
import json
import os
import time
from sklearn.metrics import classification_report
from collections import Counter
import pandas as pd
import openpyxl
# generate the bigram dictionary

def load_glove(pre_trained_txt):
    #    load the pre-trained word embeddings
    word_vecs = {}
    with open(pre_trained_txt, encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])
    return word_vecs

def get_ent_vec_char_vec(node_size, path, word_vecs):
    ent_names = json.load(
        open("translated_ent_name/"+path+".json", "r"))  # [ent_ids_1（中文）翻译后的实体名；ent_ids_2（英文）实体名]，字典类型，{id:[实体名分词]}

    '''
    d={bigram:index}，d={'ab':0,'bd':1,...,}
    '''
    d = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in d:
                    d[word[idx:idx + 2]] = count
                    count += 1

    # generate the word-level features and char-level features

    '''
        char_vec[i] :   第i个实体的二元子词出现次数向量
    '''
    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(d)))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, d[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5  # 若实体名称未出现在词表，用于生成一个0到1的随机符点数: 0 <= n < 1.0。减0.5是为了均值为0

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(d)) - 0.5  # 若字符未出现在词表，实体向量[0,1)随机生成，
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])  # 归一化
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])  # 归一化
    return ent_vec, char_vec


## %%     build the relational adjacency matrix
def get_ent_rel_vec(all_triples, ent_vec, ent_rel_vec=False):
    #   dr: dictinary of r,记录包含关系r的三元组个数{r1_id:count_1,...,ri_id:count_i,...,}
    #   neibor_rel:生成每个节点的邻居关系矩阵[头实体id:[关系id1,...,关系idi...,]]
    #   neibor_ent:生成每个关系的邻居关系矩阵[关系id:[头实体id1,...,头实体idi...,]，[尾实体id1,...,尾实体idi...,]]，不对实体进行去重
    dr = {}
    neibor_rel = {}
    neibor_ent = {}
    for x, r, y in all_triples:
        if r not in dr:
            dr[r] = 0
        dr[r] += 1

        if x not in neibor_rel:
            neibor_rel[x] = [r]
        else:
            neibor_rel[x].append(r)
        if r not in neibor_ent:
            neibor_ent[r] = [[x], [y]]
        else:
            neibor_ent[r][0].append(x)
            neibor_ent[r][1].append(y)

    if ent_rel_vec:  # 是否对实体特征增加关系模块
        # 对关系r根据TransE思想进行编码，rel_vec=[rel_vec_1;...;rel_vec_1;...]关系编码矩阵
        sample_ent = np.median(list(dr.values())).astype(int)  # 对于每个关系,采样的三元组个数为关系出现次数的中位数
        rel_vec = np.zeros((rel_size, 300))
        for i in neibor_ent.keys():  # 表示关系id
            num_ent = len(neibor_ent[i][0])  # 关系i的头实体个数
            h_list = neibor_ent[i][0]  # 关系i的头实体id，list
            t_list = neibor_ent[i][1]  # 关系i的尾实体id，list
            if num_ent <= sample_ent:
                rel_vec[i] = rel_vec[i] / np.linalg.norm(rel_vec[i])  # 行归一化,加0.1避免分母过小
            else:
                index = np.random.choice(num_ent, size=sample_ent, replace=False)  # 采样
                slice_h = tf.gather(indices=index, params=h_list).numpy()
                slice_t = tf.gather(indices=index, params=t_list).numpy()
                rel_vec[i] = np.sum(ent_vec[slice_t] - ent_vec[slice_h], axis=0) / sample_ent
            if np.sum(rel_vec[i]) == 0:  # 若关系长度为0，随机生成均值为0的向量
                rel_vec[i] = np.random.random(300) - 0.5

            rel_vec[i] = rel_vec[i] / np.linalg.norm(rel_vec[i])  # 行归一化

        # 每个实体的关系模块嵌入编码，对每个实体采样sample_ent个邻居关系，关系模块编码为其均值
        temp = []
        for rel_list_i in list(neibor_rel.values()):
            temp.append(len(rel_list_i))
        sample_rel = np.median(temp).astype(int)  # 对于每个关系,采样的三元组个数为关系出现次数的中位数,此为6
        ent_rel_vec = np.zeros((node_size, 300))
        for i in neibor_rel.keys():
            num_rel = len(neibor_rel[i])  # 实体i的关系个数
            if num_rel <= sample_rel:
                # ent_rel_vec[i] = np.sum(rel_vec[neibor_rel[i]],axis= 0 )/num_rel
                ent_rel_vec[i] = ent_rel_vec[i] / np.linalg.norm(ent_rel_vec[i])  # 行归一化
            else:
                index = np.random.choice(num_rel, size=sample_rel, replace=False)  # 采样
                slice = tf.gather(indices=index, params=neibor_rel[i]).numpy()
                # ent_rel_vec[i] = np.sum(rel_vec[slice] ,axis= 0 ) / sample_rel
                ent_rel_vec[i] = ent_rel_vec[i] / np.linalg.norm(ent_rel_vec[i] + 0.1)  # 行归一化
            if np.sum(ent_rel_vec[i]) == 0:  # 若长度为0，随机生成均值为0的向量
                ent_rel_vec[i] = np.random.random(300) - 0.5
            ent_rel_vec[i] = ent_rel_vec[i] / np.linalg.norm(ent_rel_vec[i])  # 行归一化
        return dr, neibor_rel, neibor_ent, ent_rel_vec

    else:
        return dr, neibor_rel, neibor_ent


def get_sparse_matrix_rel(node_size, all_triples,neibor_rel_count):
    sparse_rel_matrix = []

    num_edge_types= np.zeros(node_size)

    for i in range(node_size):#每个实体的关系种类数
        num_edge_types[i] =  len(list(neibor_rel_count[i].values())) + 1

    for i in range(node_size):
        sparse_rel_matrix.append([i, i,  1.0/num_edge_types[i] ])  #

    for h, r, t in all_triples:
        sparse_rel_matrix.append([h, t, 1.0/(num_edge_types[i] * neibor_rel_count[h][r] )])

    sparse_rel_matrix = np.array(
        sorted(sparse_rel_matrix, key=lambda x: x[0]))  # 将sparse_rel_matrix按第0维索引排序，即按entity_id排序
    sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:, :2], values=sparse_rel_matrix[:, 2],
                                        dense_shape=(node_size, node_size))
    return sparse_rel_matrix


def get_sparse_matrix_SEU(node_size, all_triples, dr):
    sparse_rel_matrix = []
    for i in range(node_size):
        sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)])  # 加了自环无法保证列和为1
    for h, r, t in all_triples:
        sparse_rel_matrix.append([h, t, np.log(len(all_triples) / dr[r])])  # 加了
    sparse_rel_matrix = np.array(
        sorted(sparse_rel_matrix, key=lambda x: x[0]))  # 将sparse_rel_matrix按第0维索引排序，即按entity_id排序
    sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:, :2], values=sparse_rel_matrix[:, 2],
                                        dense_shape=(node_size, node_size))
    return sparse_rel_matrix

def get_feature(ent_vec, char_vec):
    mode = "hybrid-level"

    if mode == "word-level":
        feature = ent_vec
    if mode == "char-level":
        feature = char_vec
    if mode == "hybrid-level":
        feature = np.concatenate([ent_vec, char_vec], -1)  # [ent_vec,char_vec]按列拼接
        # feature = np.concatenate([ent_vec,char_vec,ent_rel_vec],-1)     #[ent_vec,char_vec]按列拼接
    feature = tf.nn.l2_normalize(feature, axis=-1)  # 按行归一化
    # feature = (feature-np.mean(feature,axis=0))/np.std(feature,axis=0)   #按行归一化

    return feature


def cal_sims(test_pair, feature, indices_1, indices_2):
    """
    DBP KG1-KG2
    P*KG2 = KG1，KG1*KG2'
    将feature根据test_pair的id表示为上下分块矩阵，feature=[Ht;Hs]，返回值为分块矩阵的乘积Ht*Hs'
    :param test_pair: shape[N,2],[[target_entity_id, source_entity_id],...,[]]，N个已对齐实体对
    :param feature: shape[2N,l]，feature=[Ht;Hs],Hs=[h1';,,,;hN'],每行表示1个entity的嵌入向量，嵌入维度为l
    :return: shape[N，N] return = Ht*Hs'
    """
    feature_a = tf.gather(indices=np.array(indices_1), params=feature)  # KG1， target
    feature_b = tf.gather(indices=np.array(indices_2), params=feature)  # KG2，source
    return tf.matmul(feature_a, tf.transpose(feature_b, [1, 0]))


def get_sims(sparse_rel_matrix, feature, test_pair, indices_1, indices_2, pad=True):  # 补0

    # feature:size[N, m], N 两个知识图谱全部节点数，m每个节点的嵌入维度
    # 进入cal_sims函数，feature_a：Ht，size[N_test,m];  feature_b：Hs，size[N_test,m]. N_test用于测试的节点/实体数，m每个节点的嵌入维度，
    # sims：表示Ht*Hs', size[N_test, N_test],
    sims = cal_sims(test_pair, feature, indices_1, indices_2)  # 初始化相似性矩阵
    #  target:zh, source:en,
    # 原始 sigma sims
    # choose the graph depth L and feature propagation
    depth = 2
    for i in range(depth):
        feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, feature)  # AL * H
        feature = tf.nn.l2_normalize(feature, axis=-1)
        # feature = tf.nn.l2_normalize(feature,axis=-1)
        # feature = (feature-np.mean(feature,axis=0))/np.std(feature,axis=0)   #按行归一化

        # 根据test_pair编号从feature中抽取子矩阵
        sims += cal_sims(test_pair, feature, indices_1, indices_2)
    # sims: size(m,n) m行，n列,m为target G中的节点数，n为source G中的节点数，m<=n
    sims /= depth + 1
    m = sims.shape[0]  # sims行数
    n = sims.shape[1]  # sims列数
    # 补0为方阵
    if pad:
        pad = np.zeros((n - m, n))
        sims = np.concatenate((sims, pad), axis=0)  # 列不变，上下拼接
    return sims
 

def detect_dangling(sims, k, path ):
    """
    输入相似度矩阵，和topk范围，输出悬挂实体检测结果
    :param sims:
    :param k: 离群值范围
    :return: pre_label_s:
    :return: pre_label_t:
    """
    ## 行列交叉取最大
    # sims = ( sims - np.mean(sims,axis=1).reshape(len(sims),1)) /np.std(sims,axis=1).reshape(len(sims),1)   #按行归一化
    # sims的行表示源实体source，列表示目标实体target
    # sims = sims + abs(np.min(sims))
    rowmax_values = tf.reduce_max(sims, axis=1)  # 每行最大值
    rowmax_indexs = tf.argmax(sims, axis=1)  # 每行最大值索引
    colmax_values = tf.reduce_max(sims, axis=0)  # 每列最大值
    colmax_indexs = tf.argmax(sims, axis=0)  # 每列最大值索引

    #sims每列的最大值及索引
    # k3 = 26.4
    # k2 = 445
    # k = 1
    result = tf.math.top_k(tf.transpose(sims),k=k)
    values = result.values.numpy()
    indices = result.indices.numpy()

    true_label_s = np.concatenate((np.zeros(15000).astype(int), np.ones(sims.shape[0] - 15000).astype(int)))
    pre_label_s = np.zeros(len(true_label_s)).astype(int)
    true_label_t = np.concatenate((np.zeros(15000).astype(int), np.ones(sims.shape[1] - 15000).astype(int)))
    pre_label_t = np.zeros(len(true_label_t)).astype(int)

    for i, j in enumerate(rowmax_indexs):  # 第i行的最大值在第j列
        # 第j列的topk值及索引
        #第i在topk中
        if (j in indices[j]):
            continue
        else:
            pre_label_s[i] = 1
            pre_label_t[j] = 1
    # print(path)

    print('k=', k )
    print('-'*50)
    print(path,'KG2:')
    t_s = classification_report(true_label_s, pre_label_s, target_names=['待对齐实体', '悬挂实体'], digits = 4, output_dict = True)
    df = pd.DataFrame(t_s).transpose()
    # csvname = '%s %s' path
    df.to_csv("%s_k%s_KG2.csv" % ( path , k ), index=True, encoding="utf_8_sig")
    print(t_s)
    print('-'*50)
    print(path,'KG1:')
    t_t = classification_report(true_label_t, pre_label_t, target_names=['待对齐实体', '悬挂实体'], digits = 4, output_dict = True)
    df = pd.DataFrame(t_t).transpose()
    df.to_csv("%s_k%s_KG1.csv" % ( path , k ), index=True, encoding="utf_8_sig")
    print(t_t)

    return pre_label_s, pre_label_t

def detect_dangling1(sims, k, path ):
    """
    输入相似度矩阵，和topk范围，输出悬挂实体检测结果
    :param sims:
    :param k: 离群值范围
    :return: pre_label_s:
    :return: pre_label_t:
    """
    ## 行列交叉取最大
    # sims = ( sims - np.mean(sims,axis=1).reshape(len(sims),1)) /np.std(sims,axis=1).reshape(len(sims),1)   #按行归一化
    # sims的行表示源实体source，列表示目标实体target
    # sims = sims + abs(np.min(sims))
    rowmax_values = tf.reduce_max(sims, axis=1)  # 每行最大值
    rowmax_indexs = tf.argmax(sims, axis=1)  # 每行最大值索引
    colmax_values = tf.reduce_max(sims, axis=0)  # 每列最大值
    colmax_indexs = tf.argmax(sims, axis=0)  # 每列最大值索引

    #sims每列的最大值及索引
    # k3 = 26.4
    # k2 = 445
    # k = 1
    result = tf.math.top_k(tf.transpose(sims),k=k)
    values = result.values.numpy()
    indices = result.indices.numpy()
    #KG1为target，KG2为source
    true_label_1 = np.concatenate((np.zeros(15000).astype(int), np.ones(sims.shape[0] - 15000).astype(int)))
    pre_label_1 = np.zeros(len(true_label_1)).astype(int)
    true_label_2 = np.concatenate((np.zeros(15000).astype(int), np.ones(sims.shape[1] - 15000).astype(int)))
    pre_label_2 = np.zeros(len(true_label_2)).astype(int)

    for i, j in enumerate(rowmax_indexs):  # 第i行的最大值在第j列
        # 第j列的topk值及索引
        #第i在topk中
        if (j in indices[j]):
            continue
        else:
            pre_label_1[i] = 1
            pre_label_2[j] = 1
    # print(path)

    # print('k=', k )
    # print('-'*50)
    # print(path,'KG2:')
    t_1 = classification_report(true_label_1, pre_label_1, target_names=['待对齐实体', '悬挂实体'], digits = 4, output_dict = True)
    df_1 = pd.DataFrame(t_1).transpose()
    # csvname = '%s %s' path
    # df_t_s.to_csv("%s_k%s_KG2.csv" % ( path , k ), index=True, encoding="utf_8_sig")
    # print(t_s)
    # print('-'*50)
    # print(path,'KG1:')
    t_2 = classification_report(true_label_2, pre_label_2, target_names=['待对齐实体', '悬挂实体'], digits = 4, output_dict = True)
    df_2 = pd.DataFrame(t_2).transpose()
    # df_s_t.to_csv("%s_k%s_KG1.csv" % ( path , k ), index=True, encoding="utf_8_sig")
    # print(t_t)
    return pre_label_1, pre_label_2, df_1.iloc[:,:3], df_2.iloc[:,:3]
def entity_align( pre_label_1, pre_label_2, indices_1, indices_2 ):
    """

    :param pre_label_1: KG1实体类型预测结果
    :param pre_label_2: KG2实体类型预测结果
    :param indices_1: KG1实体序列
    :param indices_2: KG2实体序列
    :return: subsims：删除悬挂实体的子矩阵
            alig_ent_id_1：存放KG1中待对齐的实体id
            alig_ent_id_2：存放KG2中待对齐的实体id
    """
    ##%%
    '''
    根据悬挂实体检测的预测结果，从原始相似矩阵sims中抽取待对齐实体形成子矩阵subsims，对subsims进行匈牙利法求指派问题的解
    '''
    alig_ent_id_1 = []#alig_ent_id_1存放KG1中待对齐的实体id
    alig_ent_id_2 = []#alig_ent_id_2存放KG2中待对齐的实体id

    alig_sims_indice_1 = []#KG1中可对齐的实体id所在的sims的行位置
    alig_sims_indice_2 = []#KG2中可对齐的实体id所在的sims的列位置

    for i, j in enumerate(pre_label_1):
        if (j == 0 ):
            alig_ent_id_1.append(indices_1[i])
            alig_sims_indice_1.append(i)
    for i, j in enumerate(pre_label_2):
        if (j == 0 ):
            alig_ent_id_2.append(indices_2[i])
            alig_sims_indice_2.append(i)
    #从全部相似度矩阵中取出可对齐实体对所在的行和列，即子矩阵
    subsims = tf.gather(indices=np.array(alig_sims_indice_1), params=sims)#取sims中的alig_ent_id_1行，记为sims_tmp
    subsims = tf.gather(indices=np.array(alig_sims_indice_2), params=subsims, axis=1)#取sims_tmp中的alig_ent_id_2列

    return subsims, alig_ent_id_1, alig_ent_id_2

def hungarian_algorithm(align_dic, subsims):
    # align_dic=dict(test_pair)
    # row_id, col_id = optimize.linear_sum_assignment(sims,maximize=True)，maximize=True表示计算最大权重匹配，sims的行表示人员，列表示工作
    result_align = optimize.linear_sum_assignment(subsims, maximize=True)
    # row_id = result[0][:10000]

    col_id = result_align[1]
    c = 0
    print('Hungarian algorithm start')
    for i, j in enumerate(col_id):
        align_1 = alig_ent_id_1[i]#可对齐实体1
        align_2 = alig_ent_id_2[j]#可对齐实体1根据匈牙利算法匹配到的实体，
        if ( (align_1 in  align_dic.keys()) and ( align_dic[align_1] == align_2 )):#可对齐实体1是KG1中可对齐的，且实体对齐结果正确
            c += 1
    hits_1 = 100 * c / subsims.shape[0]
    hits_2 = 100 * c / subsims.shape[1]
    # hits =100 * c / len(col_id)
    print("Hungarian algorithm: hits@1_1 : %.2f%%" % (hits_1))
    print("Hungarian algorithm: hits@1_2 : %.2f%%" % (hits_2))
    return hits_1, hits_2


def hungarian_algorith_whole_align(align_dic, subsims):
    # align_dic=dict(test_pair)
    # row_id, col_id = optimize.linear_sum_assignment(sims,maximize=True)，maximize=True表示计算最大权重匹配，sims的行表示人员，列表示工作
    result_align = optimize.linear_sum_assignment(subsims, maximize=True)
    # row_id = result[0][:10000]

    col_id = result_align[1]
    c = 0
    print('Hungarian algorithm start')
    for i, j in enumerate(col_id):
        align_1 = alig_ent_id_1[i]#可对齐实体1
        align_2 = alig_ent_id_2[j]#可对齐实体1根据匈牙利算法匹配到的实体，
        if ( (align_1 in  align_dic.keys()) and ( align_dic[align_1] == align_2 )):#可对齐实体1是KG1中可对齐的，且实体对齐结果正确
            c += 1
    hits_1 = 100 * c / 15000
    hits_2 = 100 * c / 15000
    # hits =100 * c / len(col_id)
    print("Hungarian algorithm: hits@1_1 : %.2f%%" % (hits_1))
    print("Hungarian algorithm: hits@1_2 : %.2f%%" % (hits_2))
    return hits_1, hits_2


if __name__ == '__main__':

    seed = 12345
    np.random.seed(seed)
    # choose the GPU, "-1" represents using the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    pre_trained_txt = "glove.6B/glove.6B.300d.txt"
    word_vecs =  load_glove(pre_trained_txt)

    # klist = [27,100,200,300,400,500,600,700,800,897]
    klist = [27]

    # Alist = ['standard', 'standard_self', 'standard_circle', 'standard_self_prob',
    #          'standard_circle_prob', 'SEU', 'rel', 'standard_unnormal_laplace',
    #           'standard_laplace','rel_lap'
    #          ]
    Alist = ['rel'
             ]
    pathlist = ["dbp_zh_en", "dbp_fr_en", "dbp_ja_en"]



    for k in klist:

        for Amode in Alist:
            # writer = pd.ExcelWriter('k_%s_A_%s.xlsx' % (k, Amode))

            for i, path in enumerate(pathlist) :
                print("*"*50)
                print('Dataset',path)
                # path = "dbp_zh_en"#KG1-zh，KG2-en
                file_path = "KGs/" + path +"/"
                all_triples, node_size, rel_size = load_triples(file_path, True)
                train_pair, test_pair = load_aligned_pair(file_path, ratio=0)
                align_dic = dict(test_pair)

                ent_ids_1 = get_ent_ids(file_path, "ent_ids_1")
                ent_ids_2 = get_ent_ids(file_path, "ent_ids_2")
                ent1_in_pair = []
                ent2_in_pair = []
                for pair in test_pair:
                    ent1_in_pair.append(pair[0])
                    ent2_in_pair.append(pair[1])

                ent1_in_pair = np.array(ent1_in_pair)
                ent1_in_pair = np.unique(ent1_in_pair)

                ent2_in_pair = np.array(ent2_in_pair)
                ent2_in_pair = np.unique(ent2_in_pair)
                unmatch1 = list(set(ent_ids_1).difference(set(ent1_in_pair)))
                unmatch2 = list(set(ent_ids_2).difference(set(ent2_in_pair)))
                np.random.shuffle(unmatch1)
                np.random.shuffle(unmatch2)

                indices_1 = list(test_pair[:, 0]) + unmatch1
                indices_2 = list(test_pair[:, 1]) + unmatch2

                #   dr: dictinary of r,记录包含关系r的三元组个数{r1_id:count_1,...,ri_id:count_i,...,}
                dr = {}
                neibor_rel = {}
                neibor_ent = {}
                neibor_rel_count = {}
                for x, r, y in all_triples:
                    if r not in dr:
                        dr[r] = 0
                    dr[r] += 1

                    if x not in neibor_rel:
                        neibor_rel[x] = [r]
                    else:
                        neibor_rel[x].append(r)
                    if r not in neibor_ent:
                        neibor_ent[r] = [[x], [y]]
                    else:
                        neibor_ent[r][0].append(x)
                        neibor_ent[r][1].append(y)

                for ent in neibor_rel.keys():
                    count = Counter(neibor_rel[ent])
                    neibor_rel_count[ent] = dict(count)


                # 得到实体嵌入矩阵feature
                ent_vec, char_vec = get_ent_vec_char_vec(node_size,path, word_vecs)
                feature = get_feature(ent_vec, char_vec)


                if Amode =='SEU':
                    sparse_rel_matrix = get_sparse_matrix_SEU(node_size, all_triples, dr)
                if Amode =='rel':
                    sparse_rel_matrix = get_sparse_matrix_rel(node_size, all_triples,neibor_rel_count)
          

                # sparse_rel_matrix = get_sparse_rel_matrix_orig(node_size, all_triples, dr)
                start1 = time.time()
                sims = get_sims(sparse_rel_matrix, feature, test_pair, indices_1, indices_2, pad=False)


                #悬空实体检测
                pre_label_1, pre_label_2,df_1,df_2 = detect_dangling1(sims, k=k , path = path)
        
                datablock = np.concatenate((df_1, df_2), axis=1)
                print("k=%s A=%s %s 悬挂实体检测" % (k, Amode, path ) + '完成')

                if i == 0:
                    danglingresult = datablock
                else:
                    danglingresult = np.concatenate((danglingresult, datablock), axis=1)

                #删除悬空实体
                subsims, alig_ent_id_1, alig_ent_id_2 = entity_align( pre_label_1, pre_label_2, indices_1, indices_2 )
                end1 = time.time()
                print("悬空实体检测计算时间",end1-start1 )

                #实体对齐
                hits = hungarian_algorithm(align_dic, subsims)
                end2 = time.time()
                print("实体对齐计算时间", end2 - end1)

            # df = pd.DataFrame(danglingresult)
            # df.to_excel('k_%s_A_%s.xlsx' % (k, Amode))



