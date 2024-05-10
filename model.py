import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import itertools
import math

class MF(nn.Module):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else: 
            self.init_embedding(0)
        
    def init_embedding(self, init): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
          
    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        # preds = u_bias + i_bias

        return preds.squeeze(dim=-1)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)



class NCF(nn.Module):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.linear_1 = nn.Linear(dim*2, dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else: 
            self.init_embedding(0)
        
    def init_embedding(self, init): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
          
    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        z_emb = torch.cat([u_latent, i_latent], axis=1)
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)
        preds = self.linear_2(h1)
        # preds = self.sigmoid(preds)
        # u_bias = self.user_bias(users)
        # i_bias = self.item_bias(items)

        # preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        # preds = u_bias + i_bias

        return preds.squeeze(dim=-1)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss



class MF_ips(nn.Module):
    """
    RD module for matrix factorization.
    """
    def __init__(self, n_user, n_item, upBound, lowBound, corY, InverP, dim=40, dropout=0):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.corY = corY
        self.upBound = upBound
        self.lowBound = lowBound
        self.invP = nn.Embedding(n_item, 2)
        self.ips_hat = InverP
        self.init_embedding(None, self.ips_hat)

        
    def init_embedding(self, init, ips_hat): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        self.invP.weight = torch.nn.Parameter(ips_hat)

    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        return preds.squeeze(dim=-1)
     
    def base_model_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones_like(y_train)

        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]

        cost = loss_f(preds, y_train)
        loss = torch.sum(weight * cost)
        return loss

    def ips_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones(y_train.shape).to('cuda')
        weight[y_train == self.corY[0]] = self.invP(items)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(items)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = - torch.sum(weight * cost)
        return loss
        

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss

    def update_ips(self):
        with torch.no_grad():
            self.invP.weight.data[self.invP.weight.data>self.upBound] = self.upBound[self.invP.weight.data>self.upBound]
            self.invP.weight.data[self.invP.weight.data<self.lowBound] = self.lowBound[self.invP.weight.data<self.lowBound]


class MF_dr(nn.Module):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, upBound, lowBound, corY, InverP, dim=40, dropout=0):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.corY = corY
        self.upBound = upBound
        self.lowBound = lowBound
        self.invP = nn.Embedding(n_item, 2)
        self.ips_hat = InverP
        self.init_embedding(None, self.ips_hat)
        
    def init_embedding(self, init, ips_hat): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        self.invP.weight = torch.nn.Parameter(ips_hat)

    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        return preds.squeeze(dim=-1)

    ''' 
    def base_model_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones(y_train.shape).to('cuda')
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]

        cost = loss_f(preds, y_train)
        loss = torch.sum(weight * cost)
        return loss
    '''
    def base_model_loss(self,users,items,y_train,g_obs,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        y_hat_obs=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        y_hat_obs = y_hat_obs.squeeze(dim=-1)
        y_hat_obs_detach=torch.detach(y_hat_obs)

        e_obs=none_criterion(y_hat_obs,y_train)
        e_hat_obs=none_criterion(y_hat_obs,g_obs+y_hat_obs_detach)
        cost_obs=e_obs-e_hat_obs

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        loss_obs=torch.sum(weight*cost_obs)

        return loss_obs
    
    def imp_model_loss(self,users,items,y_train,y_hat,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        e_hat=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        e_hat = e_hat.squeeze(dim=-1)
        e = y_train - y_hat
        cost_e = none_criterion(e_hat, e)

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        return torch.sum(weight*cost_e)

    '''
    def ips_loss(self, users, items, y_train, loss_f):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        preds = preds.squeeze(dim=-1)

        weight = torch.ones(y_train.shape).to('cuda')
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]

        cost = loss_f(preds, y_train)
        #为了实现max优化loss取负了
        loss = - torch.sum(weight * cost)
        return loss
    '''
    def base_dr_loss(self,users,items,y_train,g_obs,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        y_hat_obs=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        y_hat_obs = y_hat_obs.squeeze(dim=-1)
        y_hat_obs_detach=torch.detach(y_hat_obs)

        e_obs=none_criterion(y_hat_obs,y_train)
        e_hat_obs=none_criterion(y_hat_obs,g_obs+y_hat_obs_detach)
        cost_obs=e_obs-e_hat_obs

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        loss_obs=-torch.sum(weight*cost_obs)

        return loss_obs
    
    def imp_dr_loss(self,users,items,y_train,y_hat,none_criterion):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        e_hat=torch.sum(u_latent * i_latent, dim=1, keepdim=True)  + u_bias + i_bias
        e_hat = e_hat.squeeze(dim=-1)
        e = y_train - y_hat
        cost_e = none_criterion(e_hat, e)

        weight = torch.ones(y_train.shape).to('cuda')
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP.weight[i]
        '''
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP(items)[y_train == self.corY[i],i]
        
        return -torch.sum(weight*cost_e)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss

    '''
    def update_dr(self):
        with torch.no_grad():
            for i in range(self.invP.weight.shape[0]):
                if sum(self.invP.weight.data[i]>self.upBound[i])!=0:
                    # print(f'## {i} is over up')
                    self.invP.weight[i] = self.upBound[i]
                elif sum(self.invP.weight.data[i]<self.lowBound[i])!=0:
                    # print(f'## {i} is over down')
                    self.invP.weight[i] = self.lowBound[i]
                else:
                    dsh=1
        return 0
    '''

    def update_dr(self):
        with torch.no_grad():
            self.invP.weight.data[self.invP.weight.data>self.upBound]=self.upBound[self.invP.weight.data>self.upBound]
            self.invP.weight.data[self.invP.weight.data<self.lowBound]=self.lowBound[self.invP.weight.data<self.lowBound]