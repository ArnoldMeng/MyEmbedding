#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pylab
from pylab import *
import random
import math
from decimal import *

import argparse, logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time


logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

plt.ion()

def PCA(data, K):
    # 数据标准化
    m = mean(data, axis=0) # 每列均值
    data -= m
    # 协方差矩阵
    C = cov(transpose(data))
    # 计算特征值特征向量，按降序排序
    evals, evecs = linalg.eig(C)
    indices = argsort(evals) # 返回从小到大的索引值
    indices = indices[::-1] # 反转
 
    evals = evals[indices] # 特征值从大到小排列
    evecs = evecs[:, indices] # 排列对应特征向量
    evecs_K_max = evecs[:, :K] # 取最大的前K个特征值对应的特征向量
 
    # 产生新的数据矩阵
    finaldata = dot(data, evecs_K_max)
    return finaldata

def visualize(feature,labels,t):

    colors = ["#63B8FF", "#76EE00", "#9B30FF", "#EEC900", "#FF4500", "#0000EE", "#228B22", "#8B0A50", "#EEB422", "#FA8072", "#CDAF95", "#AEEEEE", "#008B45"]
    # rewrite = open("emb/visualize.txt",'w')
    # for i in range(len(nodes)):
    #     rewrite.write(' '.join((str(f) for f in nodes[str(i)]))+'\n')
    # rewrite.close()
    # X = np.loadtxt("emb/visualize.txt")
    X = feature
    # Y = tsne.tsne(X, 2, 50, 20.0)
    Y = PCA(X,2)
    # labels = [1]*len(X)
    ldic = {}
    for i in range(len(labels)):
        if labels[i] not in ldic:
            ldic[i] = []
    # print(Y,type(Y))
    # print()
    vdic = {}
    yl = Y.tolist()

    plt.clf()
    for i in range(len(labels)):
        if labels[i] not in vdic:
            vdic[labels[i]] = [[],[]]
        vdic[labels[i]][0].append(yl[i][0])
        vdic[labels[i]][1].append(yl[i][1])
    for i in vdic:
        pylab.scatter(vdic[i][0], vdic[i][1], 80, c=colors[int(i)%len(colors)],label=str(i))
    # pc = (pylab.scatter(Y[:, 0], Y[:, 1], 80, c=labels))
    plt.legend()
    plt.pause(t)
    
    # plt.close()
    # pylab.show()
def read_label(path):
    labelx = open(path, 'r')
    labels = []
    for line in labelx.readlines()[1:]:
        labels.append(int(line.split()[-1]))
    return labels

def read_graph(path): # return a adjacent matrix
    graphx = open(path, 'r')
    edges = {}
    for line in graphx.readlines():
        ls = line.strip().split()
        for i in range(2):
            if ls[i] not in edges:
                edges[ls[i]] = set()
            edges[ls[i]].add(ls[1-i])

    return edges

def init2(edges,dim):
    initials = []
    for i in range(dim):
        tmp = [(random.random()) for d in range(dim)]
        tmpv = sum(map(lambda i:i**2,tmp)) ** 0.5
        tmp = list(map(lambda i:i/tmpv,tmp))
        if initials == []:
            initials.append(tmp)


def init(edges,dim, mode = 1):
    ngb = np.zeros((len(edges),len(edges)))
    for s in edges:
        for t in edges[s]:
            ngb[int(s)][int(t)] = 1.0

    if mode == 0: # totally random
        return (ngb,np.array([[(random.random()) for d in range(dim)] for n in range(len(edges))]))
    elif mode == 1: # continuously random with respect to degree
        initials = []
        ransn = 25
        for i in range(ransn):
            tmp = [(random.random()) for d in range(ransn)]
            # ttt = sum(list(map(lambda i: float(i)**2, tmp)))
            # print(i,ttt,sum(ttt),list(ttt))
            tmpv = sum(list(map(lambda i: float(i)**2, tmp))) ** 0.5
            tmp = list(map(lambda i: i/tmpv*(ransn-1)**0.2,tmp))
            if initials == []:
                initials.append(tmp)
            else:
                initials.append(list(map(lambda i,j:i+j,tmp,initials[-1])))
        nodes = {}
        for e in edges:
            nodes[e] = initials[len(edges[e])-1] if len(edges[e]) < ransn else initials[-1]
        feature = np.array([nodes[str(n)][:] for n in range(len(edges))])
        return (ngb,feature)



    initials = {}
    nodes = {}

    for e in edges:
        if len(edges[e]) not in initials:
            initials[len(edges[e])] = np.array([ (random.random()) for d in range(dim)])
        nodes[e] = initials[len(edges[e])]

    feature = np.array([nodes[str(n)][:] for n in range(len(edges))])
    # print(feature)
    return (ngb,feature)


def mapping(fea,ft = 1):
    if ft == 1:
        return np.sin(fea)
    elif ft == 2:
        return np.cos(fea)
    elif ft == 3:
        return np.tanh(fea)
    elif ft == 4:
        return np.arctan(fea)
    elif ft == 5:
        return np.tanh(0.5*np.around(np.sin(fea), decimals = 5))   
        


def train(edges,dim,labels):
    ngb,feature = init(edges,dim)
    # visualize(nodes,labels)
    # print([(nodes['1'][i]-nodes['37'][i]) for i in range(dim)])
    for iii in range(100):
        if iii%20 == 0:
            visualize(feature,labels,1)
        # else:
        #     visualize(feature,labels,0.1)

        feature = np.around(np.dot(ngb,mapping(feature)), decimals = 5)
    return feature       

# important!!!!
# id for each element must start from 0 and be continuous int
def main():
    # path = 'graph/barbell.edgelist'
    # labels = [0]*9 + [1,2,3,4,5,6,6,5,4,3,2,1] + [0]*9
    # edges = read_graph(path)

    # path = 'graph/karate-mirrored.edgelist'
    # edges = read_graph(path)
    # labels = [0,1] + [0]*35 + [1] + [0]*30

    names = ['brazil-airports',
                'europe-airports',
                'usa-airports']
    name = names[2]
    edges = read_graph('graph/%s.edgelist' % name)
    labels = read_label('graph/labels-%s.txt' % name)


    nodes = train(edges,128,labels)
    visualize(nodes,labels,100)




if __name__ == '__main__':
    main()