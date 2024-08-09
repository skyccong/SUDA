import numba as nb
import numpy as np
import os
import time
import multiprocessing
import tensorflow as tf
from scipy import optimize

def load_triples(file_path,reverse = True):
    @nb.njit
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            if reverse:
                reversed_triples[i,1] = triples[i,1] + rel_size
            else:
                reversed_triples[i,1] = triples[i,1]
        return reversed_triples
    
    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()
        
    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()
        
    triples = np.array([line.replace("\n","").split("\t") for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:,0]),np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:,1]) + 1
    
    all_triples = np.concatenate([triples,reverse_triples(triples)],axis=0)
    all_triples = np.unique(all_triples,axis=0)
    
    return all_triples, node_size, rel_size*2 if reverse else rel_size

def load_aligned_pair(file_path,ratio = 0.3):
    if "sup_ent_ids" not in os.listdir(file_path):
        with open(file_path + "ref_ent_ids") as f:
            aligned = f.readlines()
    else:
        with open(file_path + "ref_ent_ids") as f:
            ref = f.readlines()
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
        aligned = ref + sup
        
    aligned = np.array([line.replace("\n","").split("\t") for line in aligned]).astype(np.int64)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]

def get_ent_ids(file_path,file_name):
    ent_ids = []
    with open(file_path + file_name) as f:
        file =  f.readlines()
    ent_ids = np.array([line.split('\t')[0] for line in file ]).astype(np.int64)
    ent_ids = np.unique(ent_ids)
    return ent_ids

def evaluate(sims_matrix, batch_size= 1024):
    '''
    对相似度矩阵sims_matrix直接按相似度大小对齐
    :param sims_matrix:shape[m∑,n] sims按行展开=[a_1;...;a_n],保留非0行
    :param ans_rank:series(m),ans_rank=[ar_1,..,ar_i,...,ar_m],ar_i表示target G中第i个实体真实对应于source G中第ar_i个实体
    :param sim每行的最大值是否处于
    :param index:shape[m]，将index按照batch分块为多个ans_rank，这里index=[index_1,...,index_m]
    :return results：shape[m,2] results_i = [i,k]，表示向量a_i中第index_i位置的元素是该向量中第k大。
    '''
    results = []
    for epoch in range(len(sims_matrix) // batch_size + 1):
        sim = sims_matrix[epoch*batch_size:(epoch+1)*batch_size]   #block
        rank = tf.argsort(-sim,axis=-1)     #block，对sim每行进行降序，返回对应值的排名。默认b = tf.argsort(a,axis=-1,direction='ASCENDING',stable=False,name=None)为升序，这里对sim取负为降序
        ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(sims_matrix)))])   #series
        results.append(tf.where(tf.equal(tf.cast(rank,ans_rank.dtype),tf.tile(np.expand_dims(ans_rank,axis=1),[1,sims_matrix.shape[1]]))).numpy())
    results = np.concatenate(results,axis=0)

    return  results

def test(sims, model = 'hungarian'):

    @nb.jit(nopython = True)
    def cal(results):
        hits1,hits10,mrr = 0,0,0
        for x in results[:,1]:
            if x < 1:
                hits1 += 1
            if x < 10:
                hits10 += 1
            mrr += 1/(x + 1)
        return hits1,hits10,mrr

    if model == "sinkhorn":
        sims = tf.exp(sims * 50)
        for k in range(10):
            sims = sims / tf.reduce_sum(sims, axis=1, keepdims=True)
            sims = sims / tf.reduce_sum(sims, axis=0, keepdims=True)
        results = evaluate(sims, batch_size= 1024)
        hits1, hits10, mrr = cal(results)
        print("sinkhorn algorithm: hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (
        hits1 / len(results) * 100, hits10 / len(results) * 100, mrr / len(results) * 100))

    elif model == "sims":
        results = evaluate(sims, batch_size= 1024)
        hits1, hits10, mrr = cal(results)
        print("distance min algorithm: hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (
        hits1 / len(results) * 100, hits10 / len(results) * 100, mrr / len(results) * 100))

    else:#model = hungarian
        # row_id, col_id = optimize.linear_sum_assignment(sims,maximize=True)，maximize=True表示计算最大权重匹配，sims的行表示人员，列表示工作
        # 不是匈牙利算法，是改进的 Jonker-Volgenant 算法，也称lapjv算法）是一个比匈牙利解法更快的算法，
        result = optimize.linear_sum_assignment(sims, maximize=True)
        # row_id = result[0][:10000]
        col_id = result[1]
        c = 0
        for i, j in enumerate(col_id):
            if i == j:
                c += 1
        print("Hungarian algorithm: hits@1 : %.2f%%" % (100 * c / len(col_id)))



# def test(sims, mode="sinkhorn", batch_size=1024):
#     if mode == "sinkhorn":
#         results = []
#         for epoch in range(len(sims) // batch_size + 1):
#             sim = sims[epoch * batch_size:(epoch + 1) * batch_size]  # block
#             rank = tf.argsort(-sim,
#                               axis=-1)  # block，对sim每行进行降序，返回对应值的排名。默认b = tf.argsort(a,axis=-1,direction='ASCENDING',stable=False,name=None)为升序，这里对sim取负为降序
#             ans_rank = np.array(
#                 [i for i in range(epoch * batch_size, min((epoch + 1) * batch_size, len(sims)))])  # series
#             results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype),
#                                              tf.tile(np.expand_dims(ans_rank, axis=1), [1, len(sims)]))).numpy())
#         results = np.concatenate(results, axis=0)[:10000]
#
#         @nb.jit(nopython=True)
#         def cal(results):
#             hits1, hits10, mrr = 0, 0, 0
#             for x in results[:, 1]:
#                 if x < 1:
#                     hits1 += 1
#                 if x < 10:
#                     hits10 += 1
#                 mrr += 1 / (x + 1)
#             return hits1, hits10, mrr
#
#         hits1, hits10, mrr = cal(results)
#         print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (
#         hits1 / len(sims) * 100, hits10 / len(sims) * 100, mrr / len(sims) * 100))
#     else:
#         c = 0
#         for i, j in enumerate(sims[1]):
#             if i == j:
#                 c += 1
#         print("hits@1 : %.2f%%" % (100 * c / len(sims[0])))