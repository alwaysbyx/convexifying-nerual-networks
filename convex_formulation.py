import numpy as np
import torch
import argparse
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import product, combinations
import cvxpy as cp
import time
import random

class convex_formulation:
    def __init__(self,n,m, beta=0.1, d=1, seed=0):
        self.m = m
        self.d = d
        self.n = n
        self.beta = beta
        self.seed = seed
    
    def data_prepare_1(self):
        np.random.seed(self.seed)
        n = self.n
        m = self.m
        d = self.d
        self.X = np.random.choice(np.arange(-n,n),size=(n,d),replace=False)
        self.X_bias = np.concatenate([self.X,np.ones((n,1))],axis=1)
        self.Y = np.random.choice([-1,1],size=(n,1))
        print('data prepared:', self.X.T, self.Y.T)

    def data_prepare_2(self):
        np.random.seed(self.seed)
        n = self.n
        m = self.m
        d = self.d
        self.X = np.array([[np.random.randn()*5, np.random.randn()*5] for _ in range(n)])
        self.X_bias = np.concatenate([self.X,np.ones((n,1))],axis=1)
        self.Y = np.zeros((n,1))
        for i in range(n):
            if np.linalg.norm(self.X[i]) <= 1.5:
                self.Y[i] = -1
            elif np.linalg.norm(self.X[i]) <= 5:
                self.Y[i] = 1
            else:
                self.Y[i] = -1
        print('data prepared:', self.X.T, self.Y.T)
    
    def SGD(self, maxite=10000, seed=0, verbose=False, lr=1e-6):
        n = self.n
        batch_size = self.n//2
        loss_record = []
        torch.manual_seed(seed)
        def init_weight(m):
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight)
                m.bias.data.fill_(0.01)
        model = torch.nn.Sequential(torch.nn.Linear(self.d+1, self.m), torch.nn.ReLU(), torch.nn.Linear(self.m, 1)).double()
        model.apply(init_weight)
        X, Y = torch.from_numpy(self.X_bias).double(), torch.from_numpy(self.Y).double()
        lr = lr
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        for ite in range(maxite):
            optim.zero_grad()
            pred_y = model(X)
            reg = 0
            for param in model.parameters():
                reg += (param**2).sum()
            loss = loss_fn(pred_y, Y) + self.beta * reg
            if ite % 1000 == 1:
                if verbose: print(f'{ite} / {maxite}: {loss.item()}')
            loss_record.append(loss.detach().numpy())
            loss.backward()
            optim.step()
        return loss_record, model

    def convex_solver(self,approximate=False, P = 10000):
        n = self.n
        d = self.d
        X = self.X_bias
        I = np.eye(n)
        if approximate:
            P = P
            all_x = []
            for i in range(P):
                u = np.random.normal(0, 1, (d+1,1))
                D = np.dot(X,u)
                D[D<=0] = 0
                all_x.append(list(np.sign(D)))
            all_x = np.unique(np.array(all_x), axis=0)
            P = len(all_x)
            print(f'approximate P={P}')
            all_D = [np.zeros((n,n)) for _ in range(P)]
            for i in range(len(all_x)):
                np.fill_diagonal(all_D[i], all_x[i])
        else:
            P = 10000000
            if d==1: target = 2*n
            elif d==2: target = 2*(n+(n-1)*(n-2)//2) 
            all_x = []
            for i in range(P):
                u = np.random.normal(0, 1, (d+1,1))
                D = np.dot(X,u)
                D[D<=0] = 0
                all_x.append(list(np.sign(D)))
                tmps = []
                if i % 10000 == 0:
                    tmp = np.unique(np.array(all_x),axis=0)
                    tmps.append(len(tmp))
                    print(len(tmp))
                    if len(tmp) == target or (len(tmps) > 3 and len(tmp) == tmps[-2]):
                        break
            all_x = np.unique(np.array(all_x), axis=0)
            P = len(all_x)
            print(f'exact P={P}')
            if n == 1:
                assert P == 2*n
            all_D = [np.zeros((n,n)) for _ in range(P)]
            for i in range(len(all_x)):
                np.fill_diagonal(all_D[i], all_x[i])
        v = cp.Variable((P,d+1))
        w = cp.Variable((P,d+1))
        obj = 0
        left_term = 0
        constraints = []
        for i in range(P):
            obj +=  self.beta * (cp.sum_squares(v[i]) + cp.sum_squares(w[i]))
            left_term += all_D[i]@X@(v[i].T-w[i].T)
            constraints.append((2*all_D[i]-I)@X@v[i].T >= np.zeros(n,))
            constraints.append((2*all_D[i]-I)@X@w[i].T >= np.zeros(n,))
        obj += 1/2 * cp.sum_squares(left_term-self.Y[:,0])
        prob = cp.Problem(cp.Minimize(obj), constraints)
        result = prob.solve(solver='GUROBI')
        print(prob.value)
        return prob.value
        
    def numerical_experiments(self,maxite=10000):
        s = time.time()
        obj = self.convex_solver()
        e = time.time()
        print(e-s)
        total_loss = []
        time_cost = [e-s]
        for seed in range(10):
            s = time.time()
            loss = self.SGD(seed=seed,maxite=maxite)
            e = time.time()
            time_cost.append(e-s)
            total_loss.append(loss)
        return total_loss, obj, time_cost

class convex_formulation_classification:
    def __init__(self,n,m, beta=0.1, d=1, seed=0):
        self.m = m
        self.d = d
        self.n = n
        self.beta = beta
        self.seed = seed
    
    def SGD(self, maxite=10000, seed=0, verbose=False, lr=1e-6):
        n = self.n
        batch_size = self.n//2
        loss_record = []
        accuracy_record = []
        torch.manual_seed(seed)
        def init_weight(m):
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight)
                m.bias.data.fill_(0.01)
        model = torch.nn.Sequential(torch.nn.Linear(self.d+1, self.m), torch.nn.ReLU(), torch.nn.Linear(self.m, 1)).double()
        model.apply(init_weight)
        X, Y = torch.from_numpy(self.X_bias).double(), torch.from_numpy(self.Y).double()
        lr = lr
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        #loss_fn = torch.nn.HingeEmbeddingLoss(reduction='mean')
        for ite in range(maxite):
            optim.zero_grad()
            pred_y = model(X)
            reg = 0
            for param in model.parameters():
                reg += (param**2).sum()
            loss = loss_fn(pred_y, Y) + self.beta * reg
            if ite % 1000 == 1:
                if verbose: print(f'{ite} / {maxite}: {loss.item()}')
            prediction = model(torch.from_numpy(self.test_data))
            accuracy_record.append(self.accuracy(prediction.detach().numpy().reshape(-1), self.test_label.reshape(-1)))
            loss_record.append(loss.detach().numpy())
            loss.backward()
            optim.step()
        return loss_record, accuracy_record, model

    def accuracy(self, predict, target):
        r = 0
        for i in range(len(predict)):
            if predict[i] < 0:
                predict[i] = -1
            else:
                predict[i] = 1
            if predict[i] == target[i]:
                r += 1
        return r / len(predict)

    def convex_solver(self,approximate=False, P = 10000):
        n = self.n
        d = self.d
        X = self.X_bias
        I = np.eye(n)
        if approximate:
            P = P
            all_x = []
            for i in range(P):
                u = np.random.normal(0, 1, (d+1,1))
                D = np.dot(X,u)
                D[D<=0] = 0
                all_x.append(list(np.sign(D)))
            all_x = np.unique(np.array(all_x), axis=0)
            P = len(all_x)
            print(f'approximate P={P}')
            all_D = [np.zeros((n,n)) for _ in range(P)]
            for i in range(len(all_x)):
                np.fill_diagonal(all_D[i], all_x[i])
        else:
            P = 10000000
            if d==1: target = 2*n
            elif d==2: target = 2*(n+(n-1)*(n-2)//2) 
            all_x = []
            for i in range(P):
                u = np.random.normal(0, 1, (d+1,1))
                D = np.dot(X,u)
                D[D<=0] = 0
                all_x.append(list(np.sign(D)))
                tmps = []
                if i % 10000 == 0:
                    tmp = np.unique(np.array(all_x),axis=0)
                    tmps.append(len(tmp))
                    print(len(tmp))
                    if len(tmp) == target or (len(tmps) > 3 and len(tmp) == tmps[-2]):
                        break
            all_x = np.unique(np.array(all_x), axis=0)
            P = len(all_x)
            print(f'exact P={P}')
            if n == 1:
                assert P == 2*n
            all_D = [np.zeros((n,n)) for _ in range(P)]
            for i in range(len(all_x)):
                np.fill_diagonal(all_D[i], all_x[i])
        v = cp.Variable((P,d+1))
        w = cp.Variable((P,d+1))
        obj = 0
        left_term = 0
        constraints = []
        for i in range(P):
            obj +=  self.beta * (cp.sum_squares(v[i]) + cp.sum_squares(w[i]))
            left_term += all_D[i]@X@(v[i].T-w[i].T)
            constraints.append((2*all_D[i]-I)@X@v[i].T >= np.zeros(n,))
            constraints.append((2*all_D[i]-I)@X@w[i].T >= np.zeros(n,))
        #obj += cp.sum(cp.pos(1-cp.multiply(self.Y.reshape(-1), left_term)))/n
        obj += 1/2 * cp.sum_squares(left_term-self.Y[:,0]) / n
        prob = cp.Problem(cp.Minimize(obj), constraints)
        result = prob.solve(solver='GUROBI')
        print(prob.value)
        return prob.value
        