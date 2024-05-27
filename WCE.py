from base_ALM_const_IL import base_ALM_const_IL
from utils_cuda import setup, load_data, set_seed, load_features, stochastic_minimizer, robust_sigmoid
import torch
import numpy as np
import os


def f(w, X):
    return robust_sigmoid(X@w)

def BinaryCrossEntropy(y_true, y_pred, weights, reduce="mean"):
    y_pred = torch.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = weights[0]*(1-y_true) * torch.log(1-y_pred + 1e-7)
    term_1 = weights[1]*y_true * torch.log(y_pred + 1e-7)
    if reduce == "sum":
        return -torch.sum(term_0+term_1, axis=0)
    else:
        return -torch.mean(term_0+term_1, axis=0)


class WCE:
    def __init__(self, X, y, device):
        self.X = X
        self.y = y
        self.t = 0.5
        data_size, feature_size = X.shape
        self.w = torch.randn(feature_size).float().to(device)
        
    def set_workspace(self, ws):
        self.ws = ws
        os.makedirs(ws, exist_ok=True)
        f = open(f"{self.ws}/log.txt", "a")
        f.close()
        
    def f(self, w, X):
        return robust_sigmoid(X@w)
    
        
    def write_to_txt(self, txt):
        f = open(f"{self.ws}/log.txt", "a")
        f.write(txt+"\n")
        print(txt)
        f.close()
        
    def fit(self):
        weights = torch.tensor([torch.sum(self.y), torch.sum(1-self.y)])
        weights = weights / torch.amin(weights)
        def classificaiton_loss(w):
            return BinaryCrossEntropy(y, f(w, X), weights)
        return stochastic_minimizer(classificaiton_loss, self.w, eps=1e-2)
    
    def eval(self, pred, y):
        ## print log results
        pred = pred.cpu().numpy()
        y_cpu = y.cpu().numpy()
        precision = (pred.T@y_cpu)/float(np.sum(pred))
        recall = (pred.T@y_cpu)/float(np.sum(y_cpu))
        f1 = (2*precision*recall)/(precision+recall)
        return precision, recall, f1

def search_t_recall(opt, w, X, y, threshold=0.8):
    from bisect import bisect
    y = y.cpu().numpy()
    logit_pred = opt.f(w, X).cpu().numpy()
    candidate = sorted(set(logit_pred))
    opt_t = None
    for i, t in enumerate(candidate[::-1]):
        pred = (logit_pred >=t)
        precision = (pred.T@y)/float(np.sum(pred))
        recall = (pred.T@y)/float(np.sum(y))
        f1 = (2*precision*recall)/(precision+recall)
        if recall >= threshold:
            opt_t = candidate[i-1]
            break
        
    pred = (logit_pred >= opt_t)
    precision = (pred.T@y)/float(np.sum(pred))
    recall = (pred.T@y)/float(np.sum(y))
    f1 = (2*precision*recall)/(precision+recall)
    opt.write_to_txt(f"For FROP\tTest: precision: {precision}\t\t recall: {recall}\t\t f1-score: {f1}")
    print(f"precision, recall, f1: {precision}\t{recall}\t{f1}")
    return opt_t
    


def search_t_precision(opt, w, X, y, threshold=0.8):
    from bisect import bisect
    y = y.cpu().numpy()
    logit_pred = opt.f(w, X).cpu().numpy()
    candidate = sorted(set(logit_pred))
    highest_p, highest_recall, opt_t = 0, 0, -1
    for i, t in enumerate(candidate):
        pred = (logit_pred >=t)
        precision = (pred.T@y)/float(np.sum(pred))
        recall = (pred.T@y)/float(np.sum(y))
        f1 = (2*precision*recall)/(precision+recall)
        if precision >= threshold:
            if highest_p < threshold or (highest_p >= threshold and recall > highest_recall):
                highest_recall = recall
                opt_t = t
                highest_p = threshold
                
        elif precision < threshold and precision > highest_p:
            opt_t = t
            highest_p = precision
            highest_recall = recall

    
        
    pred = (logit_pred >= opt_t)
    precision = (pred.T@y)/float(np.sum(pred))
    recall = (pred.T@y)/float(np.sum(y))
    f1 = (2*precision*recall)/(precision+recall)
    print(f"precision, recall, f1: {precision}\t{recall}\t{f1}")
    opt.write_to_txt(f"For FPOR\tTest: precision: {precision}\t\t recall: {recall}\t\t f1-score: {f1}")
    return opt_t



def search_t_f1(opt, w, X, y):
    from bisect import bisect
    y = y.cpu().numpy()
    logit_pred = opt.f(w, X).cpu().numpy()
    candidate = sorted(set(logit_pred))
    highest_f1, opt_t = 0, -1
    for i, t in enumerate(candidate):
        pred = (logit_pred >=t)
        precision = (pred.T@y)/float(np.sum(pred))
        recall = (pred.T@y)/float(np.sum(y))
        f1 = (2*precision*recall)/(precision+recall)
        if f1 > highest_f1:
            highest_f1 = f1
            opt_t = t
        
    pred = (logit_pred >= opt_t)
    precision = (pred.T@y)/float(np.sum(pred))
    recall = (pred.T@y)/float(np.sum(y))
    f1 = (2*precision*recall)/(precision+recall)
    print(f"precision, recall, f1: {precision}\t{recall}\t{f1}")
    opt.write_to_txt(f"For OFBS\tTest: precision: {precision}\t\t recall: {recall}\t\t f1-score: {f1}")
    return opt_t


def write_to_txt(ws, txt):
    f = open(f"{ws}/log.txt", "a")
    f.write(txt+"\n")
    print(txt)
    f.close()


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
    
    opt = WCE(X, y, device)
    
    if args.decouple:
        opt.set_workspace(f"logs_decouple/{args.ds}/WCE_final/{args.seed}/")
    else:
        opt.set_workspace(f"logs/{args.ds}/WCE_final/{args.seed}/")

    if os.path.exists(f"{opt.ws}/model.npz"):
        exit("model exists")
        
    w = opt.fit()
    
    
    
    opt.write_to_txt("=========================final evaluation===============================")
    pred = (opt.f(w, X)>=opt.t).int()
    
    p, r, f1 = opt.eval(pred, y)
    opt.write_to_txt(f"Train: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
    opt_p = search_t_precision(opt, w, X, y, threshold=0.8)
    opt_r = search_t_recall(opt, w, X, y, threshold=0.8)
    opt_f = search_t_f1(opt, w, X, y)

    if args.decouple:
        X, y = load_features(ds=ds, split="test", device=device, seed=args.seed)
    else:
        X, y = load_data(ds=ds, split="test", bias=bias, device=device)
    pred = (opt.f(w, X)>=opt.t).int()
    
    p, r, f1 = opt.eval(pred, y)
    opt.write_to_txt(f"Test: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
    

    pred = (opt.f(w, X)>=opt_p).int()
    p, r, f1 = opt.eval(pred, y)
    opt.write_to_txt(f"For FPOR\tTest: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
    
    pred = (opt.f(w, X)>=opt_r).int()
    p, r, f1 = opt.eval(pred, y)
    opt.write_to_txt(f"For FROP\tTest: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
    
    
    pred = (opt.f(w, X)>=opt_f).int()
    p, r, f1 = opt.eval(pred, y)
    opt.write_to_txt(f"For OFBS\tTest: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
    
    np.savez(f"{opt.ws}/model.npz", w=w.cpu().numpy())
