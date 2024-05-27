from base_ALM_const_IL import base_ALM_const_IL
from utils_cuda import setup, load_data, set_seed, stochastic_minimizer, BinaryCrossEntropy, load_features
import torch
import numpy as np
from copy import deepcopy
import os


class OFBS(base_ALM_const_IL):
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

        n_constraints = data_size if not Folding else 1
        
        # lam = np.random.randn(n_constraints)
        self.lam = torch.zeros(n_constraints, 1).to(device)
        self.rhos = torch.ones(n_constraints, 1).to(device)
        self.min_rho = 1
        self.max_iter = 100
        self.reg_weights = 1
    
    def objective(self, s, y):
        beta = 1
        # return -s.T@y/(s.T@(y==0).float()+torch.sum((y).float())*beta**2)
        return -(1+beta**2)*s.T@y/(torch.sum(s)+torch.sum((y).float())*beta**2)
    
    def eval(self, pred, y):
        ## print log results
        pred = pred.float()
        y = y.float()
        beta = 1
        return (1+beta**2)*pred.T@y/(torch.sum(pred)+torch.sum((y).float())*beta**2)

    ## FPOR
    def lagrangian(self, X, y, w, s, t, lam, rhos, data_size, alpha, Folding=True):
        fx = self.f(w, X)
        i_c = self.indicator_constr(s, y, fx, t, data_size, Folding=Folding)
        ## revise s with error correction
        obj = self.objective(s, y)
        # reg_term = self.reg_weights*fx.T@(1-fx)
        reg_term = self.reg_weights*(BinaryCrossEntropy(s, fx))
        return obj + lam.T@i_c + 0.5*torch.sum(rhos * (i_c** 2))  + reg_term


    def update_langrangian_multiplier(self):
        ## update multiplier
        data_size = self.get_data_size()
        fx = self.f(self.w, self.X)
        i_c = self.indicator_constr(self.s, self.y, fx, self.t, data_size, Folding=self.Folding)
        self.lam += self.rhos*i_c
    
    
    def log(self, cur_iter):
        ## print log results
        fx = self.f(self.w, self.X)
        data_size = self.get_data_size()
        y_cpu = self.y.cpu().numpy()
        s_cpu = self.s.cpu().numpy()
        pred = (fx>=self.t).cpu().numpy().astype(int)
        i_c = self.indicator_constr(self.s, self.y, fx, self.t, data_size, Folding=self.Folding).cpu().numpy()
        lam_cpu = self.lam[:data_size].cpu().numpy()
        
        self.write_to_txt(f"========={cur_iter}/{self.max_iter}===============")
        self.write_to_txt(f"lambda: {np.min(lam_cpu)}, {np.max(lam_cpu)}, {lam_cpu[-1]}")
        self.write_to_txt(f"violation: {np.min(abs(i_c))}, {np.max(abs(i_c))}")
        self.write_to_txt(f"rho: {torch.amin(self.rhos)}, {torch.amax(self.rhos)}")
        pred = (self.f(self.w, self.X)>=self.t).int()
        r_obj = self.eval(pred, self.y)
        self.write_to_txt(f"real obj: {r_obj}")
        e_obj = self.eval(self.s, self.y)
        self.write_to_txt(f"estimated obj: {e_obj}")
        
        return r_obj, i_c
    
    def ALM(self):
        if self.ws:
            self.init_warm_start()
        obj, i_c = self.log(-1)
        best_w = self.w
        best_obj = 0
        data_size = self.get_data_size()
        pre_constr = np.zeros(data_size+1) if not self.Folding else np.zeros(2)
        min_rho = 1
        
        for cur_iter in range(self.max_iter):
            self.reg_weights = cur_iter**1
            
            self.w.data = stochastic_minimizer(self.lagrangian_helper_w, self.w, eps=0.1).data
            self.write_to_txt("\n\n")
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
    if args.decouple:
        X, y = load_features(ds=ds, split="train", device=device, seed=args.seed)
    else:
        X, y = load_data(ds=ds, split="train", bias=bias, device=device)
    
    opt = OFBS(X, y, args.alpha, device, Folding=True, ws=args.ws)
    if args.decouple:
        opt.set_workspace(f"logs_decouple/{args.ds}/OFBS/{args.seed}/")
    else:
        opt.set_workspace(f"logs/{args.ds}/OFBS/{args.seed}/")
    if os.path.exists(f"{opt.workspace}/model.npz"):
        exit("model exists")
    w = opt.ALM()
    
    opt.write_to_txt("=========================final evaluation===============================")
    pred = (opt.f(w, X)>=opt.t).int()
    
    obj = opt.eval(pred, y)
    opt.write_to_txt(f"Train: real obj: {obj}")

    if args.decouple:
        X, y = load_features(ds=ds, split="test", device=device, seed=args.seed)
    else:
        X, y = load_data(ds=ds, split="test", bias=bias, device=device)
    pred = (opt.f(w, X)>=opt.t).int()
    obj = opt.eval(pred, y)
    opt.write_to_txt(f"Test: real obj: {obj}")
    np.savez(f"{opt.workspace}/model.npz", w=w.cpu().numpy(), s=opt.s.cpu().numpy())
