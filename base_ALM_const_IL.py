import numpy as np
import torch
from utils_cuda import warm_start, BinaryCrossEntropy, stochastic_minimizer, robust_sigmoid
from copy import deepcopy
import os

class base_ALM_const_IL:
    def __init__(self, X, y, alpha, device, Folding=True, ws=True):
        self.ws = ws
        self.X = X
        self.y = y.reshape(-1, 1)
        self.Folding = Folding
        data_size, feature_size = X.shape[0], X.shape[1]
        
        self.w = torch.randn(feature_size, 1).float().to(device)
        self.w /= torch.norm(self.w, p=2)
        self.alpha = alpha
        self.t = 0.5
        self.s = torch.rand(data_size, 1).to(device)

        n_constraints = data_size+1 if not Folding else 2
        
        # lam = np.random.randn(n_constraints)
        self.lam = torch.zeros(n_constraints, 1).to(device)
        self.rhos = torch.ones(n_constraints, 1).to(device) * 100
        self.min_rho = 1
        self.max_iter = 100
        self.reg_weights = 1
        
    def set_workspace(self, ws):
        self.workspace = ws
        os.makedirs(ws, exist_ok=True)
        f = open(f"{self.workspace}/log.txt", "a")
        f.close()
        
    def write_to_txt(self, txt):
        f = open(f"{self.workspace}/log.txt", "a")
        f.write(txt+"\n")
        print(txt)
        f.close()
    
    def f(self, w, X):
        return robust_sigmoid(X@w)
    
    def get_data_size(self):
        return self.X.shape[0]
    
    def get_feature_size(self):
        return self.X.shape[1]
        
        
    def init_warm_start(self):
        self.w.data = warm_start(self.w, self.X, self.y)
        self.s.data = (self.f(self.w, self.X)>self.t).float().data
        
    def objective(self):
        return None

    def metric_constr(self):
        return None
    
    def indicator_constr(self, s, y, fx, t, data_size, ineq=True, Folding=False, smooth=False, normalize=True):
        all_constr = torch.zeros(data_size, 1).to(s.device)
        
        weights = torch.tensor([torch.sum(y==1), torch.sum(y==0)])
        weights = weights / float(y.shape[0])
        
        ## negative
        idx = torch.where(y==0)[0]
        n_tmp = torch.maximum(s[idx]+fx[idx]-t-1, torch.tensor(0)) - torch.maximum(-s[idx], fx[idx]-t)
        # n_tmp = n_tmp/np.maximum(abs(np.maximum(-s[idx], fx[idx]-t)), 1e-9) if normalize else n_tmp
        all_constr[idx] = torch.maximum(-n_tmp, torch.tensor(0)) if ineq else n_tmp
        # all_constr[idx] = np.log(1+n_tmp**2)

        if smooth:
            all_constr[idx] = all_constr[idx] ** 2
        all_constr[idx] = weights[0] * all_constr[idx]
        
        ## positive
        idx = torch.where(y==1)[0]  
        p_tmp = torch.maximum(s[idx]+fx[idx]-t-1, torch.tensor(0)) - torch.maximum(-s[idx], fx[idx]-t)
        # p_tmp = p_tmp/np.maximum(abs(np.maximum(-s[idx], fx[idx]-t)), 1e-9) if normalize else p_tmp

        all_constr[idx] = torch.maximum(p_tmp, torch.tensor(0)) if ineq else p_tmp
        # all_constr[idx] = np.log(1+p_tmp**2)
        
        if smooth:
            all_constr[idx] = all_constr[idx] ** 2
        all_constr[idx] = weights[0] * all_constr[idx]


        return torch.mean(all_constr).reshape(1, ) if Folding else all_constr
        # return torch.mean(torch.topk(all_constr, k=int(data_size*0.25))[0]).reshape(1, ) if Folding else all_constr
        

    
    
    ## FPOR
    def lagrangian(self, X, y, w, s, t, lam, rhos, data_size, alpha, Folding=True):
        fx = self.f(w, X)
        i_c = self.indicator_constr(s, y, fx, t, data_size, Folding=Folding)
        # mask = (i_c == 0).astype(int)
        mask = None
        m_c = self.metric_constr(s, y, alpha).reshape(1, )
        ## revise s with error correction
        obj = self.objective(s, y)
        a_c = torch.concat([i_c, m_c]).reshape(-1, 1)
        # reg_term = reg_weights*fx.T@(1-fx)
        reg_term = self.reg_weights*(BinaryCrossEntropy(s, fx))
        return obj + lam.T@a_c + 0.5*torch.sum(rhos * (a_c** 2))  + reg_term


    def lagrangian_helper_s(self, my_s):
        data_size = self.get_data_size()
        return self.lagrangian(self.X, self.y, self.w, my_s, self.t, self.lam, self.rhos, data_size, self.alpha, self.Folding)

    def lagrangian_helper_w(self, my_w):
        data_size = self.get_data_size()
        return self.lagrangian(self.X, self.y, my_w, self.s, self.t, self.lam, self.rhos, data_size, self.alpha, self.Folding)

    def update_langrangian_multiplier(self):
        ## update multiplier
        data_size = self.get_data_size()
        fx = self.f(self.w, self.X)
        i_c = self.indicator_constr(self.s, self.y, fx, self.t, data_size, Folding=self.Folding)
        m_c = self.metric_constr(self.s, self.y, self.alpha).reshape(1, )
        a_c = torch.concat([i_c, m_c]).reshape(-1, 1)
        self.lam += self.rhos*a_c.data
    
    def ALM(self):
        if self.ws:
            self.init_warm_start()
        best_w = None
        best_obj = 0
        best_vio = 1
        data_size = self.get_data_size()
        pre_constr = np.zeros(data_size+1) if not self.Folding else np.zeros(2)
        min_rho = 1
        
        for cur_iter in range(self.max_iter):
            self.reg_weights = cur_iter ** 2
            
            self.w.data = stochastic_minimizer(self.lagrangian_helper_w, self.w, eps=0.1).data
            
            self.write_to_txt("\n\n")
            self.s.data = stochastic_minimizer(self.lagrangian_helper_s, self.s, bound=[0, 1], eps=1).data
            
            self.update_langrangian_multiplier()
            
            obj, const, a_c = self.log(cur_iter)
            cur_vio = np.amax([self.alpha - const, 0])
        
            if cur_vio < best_vio: ## last element in a_c is metric violation value
                self.write_to_txt(f"case 1: {cur_vio}\t{best_vio}")
                self.write_to_txt(f"{self.alpha} \t {const}")
                best_obj = obj
                best_w = deepcopy(self.w)
                best_vio = cur_vio
            elif cur_vio == best_vio and best_obj < obj:
                self.write_to_txt(f"case 2: {obj}\t{best_obj}")
                best_obj = obj
                best_w = deepcopy(self.w) 
            else:
                self.write_to_txt("case 3")
                pass


            for idx, (p, a) in enumerate(zip(pre_constr, a_c)):
                # if idx < 2*data_size:
                if abs(p) < abs(a):
                    self.rhos[idx] *= 10
                elif abs(p) == abs(a):
                    self.rhos[idx] = self.rhos[idx]
                else:
                    self.rhos[idx] /= 2
                self.rhos[idx] = max(min_rho, self.rhos[idx])
            pre_constr = a_c
        return best_w
        
        
if __name__ == "__main__":
    pass
    # args = setup()
    # ds = args.ds
    # set_seed(args.seed)
    # X, y = load_data(ds=ds, split="train", bias=bias, device=device)
    