from utils.base_ALM_const_IL import base_ALM_const_IL
from utils.utils_cuda import setup, load_data, set_seed, load_features
import torch
import numpy as np
import os



g_objs = []
g_const = []
g_objs_s = []
g_const_s = []


class FROP(base_ALM_const_IL):
    def objective(self, s, y):
        return -(s.T@y) / torch.sum(s)

    ## precision
    def metric_constr(self, s, y, alpha):
        return torch.maximum(torch.tensor(0), alpha -(s.T@y) / torch.sum(y))

    
    def eval(self, pred, y):
        ## print log results
        pred = pred.cpu().numpy()
        y_cpu = y.cpu().numpy()
        obj = (pred.T@y_cpu)/float(np.sum(pred))
        const = (pred.T@y_cpu)/float(np.sum(y_cpu))
        return obj, const
    
    def log(self, cur_iter):
        ## print log results
        fx = self.f(self.w, self.X)
        data_size = self.get_data_size()
        y_cpu = self.y.cpu().numpy()
        s_cpu = self.s.cpu().numpy()
        pred = (fx>=self.t).cpu().numpy().astype(int)
        i_c = self.indicator_constr(self.s, self.y, fx, self.t, data_size, Folding=self.Folding).cpu().numpy()
        m_c = self.metric_constr(self.s, self.y, self.alpha).reshape(1, ).cpu().numpy()
        lam_cpu = self.lam[:data_size].cpu().numpy()
        
        self.write_to_txt(f"========={cur_iter}/{self.max_iter}===============")
        self.write_to_txt(f"lambda: {np.min(lam_cpu)}, {np.max(lam_cpu)}, {lam_cpu[-1]}")
        self.write_to_txt(f"violation: {np.min(abs(i_c))}, {np.max(abs(i_c))}, {m_c}")
        pred = (self.f(self.w, self.X)>=self.t).int()
        r_obj, r_const = self.eval(pred, self.y)
        self.write_to_txt(f"real obj: {r_obj} \t\t const: {r_const}")
        e_obj, e_const = self.eval(self.s, self.y)
        self.write_to_txt(f"estimated obj: {e_obj}\t\t const: {e_const}")


        # r_obj_s, r_const_s = self.eval(fx, self.y)
        g_objs.append(r_obj)
        g_const.append(r_const)
        g_objs_s.append(e_obj)
        g_const_s.append(e_const)

        
        a_c = np.concatenate([i_c, m_c])
        return r_obj, r_const, a_c
    
if __name__ == "__main__":
    args = setup()
    set_seed(args.seed)
    bias = True
    ds = args.ds
    device = torch.device("cuda")
    X, y = load_data(ds=ds, split="train", bias=bias, device=device)
    
    opt = FROP(X, y, args.alpha, device, Folding=True, ws=args.ws)
    opt.set_workspace(f"logs/{args.ds}/FROP/{args.seed}/")
    # if os.path.exists(f"{opt.workspace}/model.npz"):
    #     exit("model exists")
    w = opt.ALM()
    
    opt.write_to_txt("=========================final evaluation===============================")
    pred = (opt.f(w, X)>=opt.t).int()
    
    obj, const = opt.eval(pred, y)
    opt.write_to_txt(f"Train: real obj: {obj}\t\t const: {const}")

    if args.decouple:
        X, y = load_features(ds=ds, split="test", device=device, seed=args.seed)
    else:
        X, y = load_data(ds=ds, split="test", bias=bias, device=device)

    pred = (opt.f(w, X)>=opt.t).int()
    obj, const = opt.eval(pred, y)
    opt.write_to_txt(f"Test: real obj: {obj}\t\t const: {const}")
    
    # np.savez(f"{opt.workspace}/model.npz", w=w.cpu().numpy(), s=opt.s.cpu().numpy())
    np.savez(f"{opt.workspace}/model.npz", w=w.cpu().numpy(), g_objs=g_objs, g_const=g_const, g_objs_s=g_objs_s, g_const_s=g_const_s)

