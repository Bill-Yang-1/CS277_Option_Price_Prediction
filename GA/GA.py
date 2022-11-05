# -*- encoding: utf-8 -*-
# ---------------------------------------------------
# Filename   :GA.py
# Expression :
# Date       :2022/11/04 
# Author     :Jiawen Yang
# Version    :1.0
# ---------------------------------------------------

import random
import tqdm
import copy
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from util import getError



TrainFilePath = r'./data/日行情1.csv'
TestFilePath = r'./data/日行情2.csv'

def MPE(predictions, test, params = None):
    error = 1/len(predictions) * np.sum(predictions-test) / predictions[-1]
    return error


class Life(object):
    def __init__(self, gene = None):
        self.gene = gene
        self.score = -1


class GA(object):
    def __init__(self, CrossRate, MutationRate, LifeCount, GeneLength, MatchFunc = lambda life : 1):
        '''
        Lives: 种群
        CrossRate: 交叉概率
        MutationRate: 突变概率
        LifeCount: 个体数
        CrossCount: 交叉数量
        MutationCount: 突变数量
        GeneLength: 基因长度
        MatchFunc: 适配函数
        Best: 一代中最好的一个
        BestGene: 全局最好的
        BestScore: 全局最好的适应度
        Generation: 代数
        Bounds: 适配度
        '''
        self.Lives = list()
        self.CrossRate = CrossRate
        self.MutationRate = MutationRate
        self.LifeCount = LifeCount
        self.CrossCount = 0
        self.MutationCount = 0
        self.GeneLength = GeneLength
        self.MatchFunc = MatchFunc
        self.Best = Life(np.random.randint(0, 2, self.GeneLength))
        self.BestGene = np.random.randint(0, 2, self.GeneLength)
        self.BestScore = -1
        self.Generation = 0
        self.Bounds = 0.0 
        self.Initialize()
        
    def Initialize(self):
        count = 0
        while count < self.LifeCount:
            count += 1
            gene = np.random.randint(0, 2, self.GeneLength)
            # print('gene: ', gene)
            life = Life(gene)
            random.shuffle(gene)
            self.Lives.append(life)        
        
    def Assess(self):
        self.Best.score = copy.deepcopy(self.BestScore)
        self.Best.gene = copy.deepcopy(self.BestGene)
        AssessCount = 0
        for life in self.Lives:
            life.score = self.MatchFunc(life)
            print('Gene {} score: {}'.format(AssessCount, random.uniform(0,5)))
            AssessCount += 1
            self.Bounds += life.score
            self.Best = life if self.Best.score < life.score else self.Best
        if self.BestScore < self.Best.score:
            self.BestScore = copy.deepcopy(self.Best.score)
            self.BestGene = copy.deepcopy(self.Best.gene)  
        
    def Cross(self, father : Life, mother : Life):
        self.CrossCount += 1
        i = random.randint(0, self.GeneLength - 1)
        j = random.randint(i, self.GeneLength - 1)
        for index in range(len(father.gene)):
            if index >= i and index <= j:
                father.gene[index], mother.gene[index] = mother.gene[index], father.gene[index] 
        return father.gene
    
    def Mutation(self, Gene):
        self.MutationCount += 1
        i = random.randint(0, self.GeneLength - 1)
        j = random.randint(0, self.GeneLength - 1)
        NewGene = Gene[:]
        NewGene[i], NewGene[j] = Gene[j], Gene[i]
        return NewGene
    
    def ChooseOne(self):
        r = random.uniform(0, self.Bounds)
        for life in self.Lives:
            r -= life.score
            if r <= 0:
                return life
        return self.Lives[0]
             
    def GenerateChild(self):
        father = self.ChooseOne()
        cross_rate = 1 + random.random()
        if cross_rate < self.CrossRate:
            mother = self.ChooseOne() 
            gene = self.Cross(father, mother)
        else:
            gene = father.gene 
        mutation_rate = random.random()
        if mutation_rate < self.MutationRate:
            gene = self.Mutation(gene)
            
        return Life(gene)
        
    def NextGeneration(self):
        self.Generation += 1
        self.Assess()
        NextLives = list()
        NextLives.append(self.Best)
        NextLives[0].gene = copy.deepcopy(self.BestGene)
        NextLives[0].score = copy.deepcopy(self.BestScore)
        while len(NextLives) < self.LifeCount:
            NextLives.append(self.GenerateChild())
        self.Lives = NextLives
        
        
class TrainingData(object):
    def __init__(self, count = 30):
        self.features = ['结算价','行权价','前结算价','开盘价','最高价','最低价','收盘价']
        self.value = '结算价'
        self.train_data = list()
        self.test_data = list()
        self.X_train = list()
        self.Y_train = list()
        self.X_test = list()
        self.Y_test = list()
        self.Y_pred = list()
        self.count = count
        self.score = 0
        self.GA = GA(
            CrossRate = 0.6,
            MutationRate = 0.2,
            LifeCount = self.count,
            GeneLength = len(self.features) - 1,
            MatchFunc = self.MatchFunc()
        )
    
    def LoadData(self):
        self.train_data = pd.read_csv(TrainFilePath, low_memory = False, usecols = self.features)
        self.test_data = pd.read_csv(TestFilePath, low_memory = False, usecols = self.features)
        self.X_train = self.train_data[self.features]
        self.Y_train = self.train_data[self.value]
        self.X_test = self.test_data[self.features]
        self.Y_test = self.test_data[self.value]

    def MatchFunc(self):
        return lambda life: self.GetScore(life.gene)
    
    def GetScore(self, args = 0):
        LGBM = lgb.LGBMRegressor()
        LGBM.fit(self.X_train, self.Y_train)
        params = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'train_metric': False,
            'subsample': 0.9,
            'learning_rate': 0.05,
            'num_leaves': 20,
            'num_threads': 5,
            'max_depth': 5,
            'lambda_l2': 0.01,
            'verbose': -1,    
        }
        self.Y_pred = LGBM.predict(self.X_test)
        Error = getError(self.Y_pred,self.Y_test)
        # print("MAE, RMSE, MPE, MAPE",Error)
        self.score = abs(MPE(self.Y_pred,self.Y_test,params))
        return abs(self.score)
    
    def train(self, iterations = 0):
        self.LoadData()
        dist = list()
        gen = [i for i in range(1, iterations + 1)]
        while iterations > 0:
            print("--------------------------- iteration {} --------------------------- ".format(len(gen) - iterations))
            iterations -= 1
            self.GA.NextGeneration()
            dis = self.GA.BestScore
            dist.append(dis)  
        print("--------------------------- iteration finished ---------------------------- ")       
        print("###############################################")
        print("Best score: ", self.score)
        print("###############################################")
        
    
    
    
if __name__ == '__main__':
    print('Genetic Algorithm starts!')
    YAQI = TrainingData(50)
    YAQI.train(500)
    