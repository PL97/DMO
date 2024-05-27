from base_ALM_const_IL import base_ALM_const_IL
from utils_cuda import setup, warm_start, load_data, set_seed, stochastic_minimizer, BinaryCrossEntropy
import torch
import numpy as np
from copy import deepcopy


class OAP(base_ALM_const_IL):
    def __init__(self, X, y, alpha, device, Folding=True):
        self.X = X
        self.y = y
        self.Folding = Folding
        self.t = 0.5
        data_size, feature_size = X.shape[0], X.shape[1]
        
        self.w = torch.randn(feature_size).float().to(device)
        self.s = torch.zeros(data_size, data_size).to(device)

        n_constraints = data_size**2 if not Folding else 1
        
        # lam = np.random.randn(n_constraints)
        self.lam = torch.zeros(n_constraints).to(device)
        self.rhos = torch.ones(n_constraints).to(device)
        self.min_rho = 1
        self.max_iter = 50
        self.reg_weights = 1
        
    def indicator_constr(self, s, y, fx, t, data_size, ineq=True, Folding=False, smooth=False, normalize=True):
        
        
        weights = torch.tensor([torch.sum(y==1), torch.sum(y==0)])
        weights = weights / float(y.shape[0])
        
        pos_ids = torch.where(y==1)[0]
        neg_ids = (y==0).flatten()
        
        
        all_const = []
        for pi in pos_ids:
            ## negative
            pn_tmp = torch.maximum(s[pi, neg_ids]+fx[neg_ids]-fx[pi]-1, torch.tensor(0)) - torch.maximum(-s[pi, neg_ids], fx[neg_ids]-fx[pi])
            pn_tmp = torch.maximum(-pn_tmp, torch.tensor(0)) if ineq else pn_tmp

            if smooth:
                pn_tmp = pn_tmp ** 2
            all_const.extend(pn_tmp)
            
            ## positive
            pp_tmp = torch.maximum(s[pi, pos_ids]+fx[pi]-fx[pos_ids]-1, torch.tensor(0)) - torch.maximum(-s[pi, pos_ids], fx[pi]-fx[pos_ids]-t)
            pp_tmp = torch.maximum(pp_tmp, torch.tensor(0)) if ineq else pp_tmp
        
            if smooth:
                pp_tmp = pp_tmp ** 2
            all_const.extend(pp_tmp)

        all_const = torch.stack(all_const)
        return torch.mean(all_const).reshape(1, ) if Folding else all_const
    
    def objective(self, s, y):
        n_pos = torch.sum(y==1)
        n_negs = torch.sum(y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(s.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        reweights = torch.ones(X.shape[0], 1).to(s.device)
        reweights[y==1] = weights[1]
        
        ret = 0
        pos_ids = torch.where(y==1)[0]
        for i in pos_ids:
            nominator, denominator = 0, 0
            if y[i] == 0:
                continue
            nominator = torch.sum(s[i, (y==1).flatten()])
            denominator = torch.sum(s[i, :])
            ret += (nominator/denominator)
        obj = (1/n_pos)*ret
        return obj
    
    def eval(self, pred, y):
        ## print log results
        pred = pred.cpu().numpy()
        y_cpu = y.cpu().numpy()
        obj = (pred.T@y_cpu)/float(np.sum(pred))
        return obj

    ## FPOR
    def lagrangian(self, X, y, w, s, t, lam, rhos, data_size, Folding=True):
        fx = self.f(w, X)
        i_c = self.indicator_constr(s, y, fx, t, data_size, Folding=Folding)
        ## revise s with error correction
        obj = self.objective(s, y)
        reg_term = self.reg_weights*fx.T@(1-fx)
        # reg_term = self.reg_weights*(BinaryCrossEntropy(s, fx))
        return obj + lam.T@i_c + 0.5*torch.sum(rhos * (i_c** 2)) + reg_term


    def update_langrangian_multiplier(self):
        ## update multiplier
        data_size = self.get_data_size()
        fx = self.f(self.w, self.X)
        i_c = self.indicator_constr(self.s, self.y, fx, self.t, data_size, Folding=self.Folding)
        self.lam += self.rhos*i_c
        
    def lagrangian_helper_s(self, my_s):
        data_size = self.get_data_size()
        return self.lagrangian(self.X, self.y, self.w, my_s, self.t, self.lam, self.rhos, data_size, self.Folding)

    def lagrangian_helper_w(self, my_w):
        data_size = self.get_data_size()
        return self.lagrangian(self.X, self.y, my_w, self.s, self.t, self.lam, self.rhos, data_size, self.Folding)

    
    
    def log(self, cur_iter):
        ## print log results
        fx = self.f(self.w, self.X)
        data_size = self.get_data_size()
        y_cpu = self.y.cpu().numpy()
        s_cpu = self.s.cpu().numpy()
        pred = (fx>=self.t).cpu().numpy().astype(int)
        i_c = self.indicator_constr(self.s, self.y, fx, self.t, data_size, Folding=self.Folding).cpu().numpy()
        lam_cpu = self.lam[:data_size].cpu().numpy()
        
        print(f"========={cur_iter}/{self.max_iter}===============")
        print(f"lambda: {np.min(lam_cpu)}, {np.max(lam_cpu)}, {lam_cpu[-1]}")
        print(f"violation: {np.min(abs(i_c))}, {np.max(abs(i_c))}")
        pred = (self.f(self.w, self.X)>=self.t).int()
        r_obj = self.eval(pred, self.y)
        print(f"real obj: {r_obj}")
        e_obj = self.eval(self.s, self.y)
        print(f"estimated obj: {e_obj}")
        
        return r_obj, i_c
    
    def warm_start(self):
        self.w.data = warm_start(self.w, self.X, self.y)
        data_size = self.get_data_size()
        fx = self.f(self.w, self.X)
        for i in range(data_size):
            for j in range(data_size):
                self.s[i, j].data = (fx[i]>fx[j]).data.float()
    
    def ALM(self):
        data_size = self.get_data_size()
        self.warm_start()
        self.s = torch.zeros(data_size, data_size).to(device)
        best_w = None
        best_obj = 0
        pre_constr = np.zeros(data_size+1) if not self.Folding else np.zeros(2)
        min_rho = 1
        
        for cur_iter in range(self.max_iter):
            self.reg_weights = cur_iter ** 2
            self.w.data = stochastic_minimizer(self.lagrangian_helper_w, self.w, eps=0.1).data
            print("\n\n")
            self.s.data = stochastic_minimizer(self.lagrangian_helper_s, self.s, bound=[0, 1], eps=1).data
            
            self.update_langrangian_multiplier()
            
            obj, i_c = self.log(cur_iter)
        
            if best_obj < obj:
                best_obj = obj
                best_w = deepcopy(self.w) 


            for idx, (p, a) in enumerate(zip(pre_constr, i_c)):
                # if idx < 2*data_size:
                if abs(p) < abs(a):
                    self.rhos[idx] *= 10
                elif abs(p) == abs(a):
                    self.rhos[idx] = self.rhos[idx]
                else:
                    self.rhos[idx] /= 2
                self.rhos[idx] = max(min_rho, self.rhos[idx])
            pre_constr = i_c
        
        return best_w
    
    
if __name__ == "__main__":
    args = setup()
    set_seed(args.seed)
    bias = True
    ds = args.ds
    device = torch.device("cuda")
    X, y = load_data(ds=ds, split="train", bias=bias, device=device)
    
    opt = OAP(X, y, args.alpha, device, Folding=True)
    w = opt.ALM()
    
    print("=========================final evaluation===============================")
    pred = (opt.f(w, X)>=opt.t).int()
    
    obj = opt.eval(pred, y)
    print(f"Train: real obj: {obj}")

    X, y = load_data(ds=ds, split="test", bias=bias, device=device)
    pred = (opt.f(w, X)>=opt.t).int()
    obj = opt.eval(pred, y)
    print(f"Test: real obj: {obj}")
