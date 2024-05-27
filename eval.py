from utils.base_ALM_const_IL import base_ALM_const_IL
from utils.utils_cuda import setup, load_data, set_seed, load_features
import torch
import numpy as np
from utils_cuda import robust_sigmoid

def search_t_recall(func, w, X, y, threshold=0.8):
    from bisect import bisect
    y = y.cpu().numpy()
    logit_pred = func(w, X).cpu().numpy()
    candidate = sorted(set(logit_pred))
    opt_t = None
    for i, t in enumerate(candidate):
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
    # opt.write_to_txt(f"For FROP\tTest: precision: {precision}\t\t recall: {recall}\t\t f1-score: {f1}")
    # print(f"precision, recall, f1: {precision}\t{recall}\t{f1}")
    return opt_t
    


def search_t_precision(opt, w, X, y, threshold=0.8):
    from bisect import bisect
    y = y.cpu().numpy()
    logit_pred = func(w, X).cpu().numpy()
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
    # print(f"precision, recall, f1: {precision}\t{recall}\t{f1}")
    # print(f"For FPOR\tTest: precision: {precision}\t\t recall: {recall}\t\t f1-score: {f1}")
    return opt_t



def search_t_f1(opt, w, X, y):
    from bisect import bisect
    y = y.cpu().numpy()
    logit_pred = func(w, X).cpu().numpy()
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
    # print(f"precision, recall, f1: {precision}\t{recall}\t{f1}")
    # opt.write_to_txt(f"For OFBS\tTest: precision: {precision}\t\t recall: {recall}\t\t f1-score: {f1}")
    return opt_t


def eval(pred, y):
    ## print log results
    pred = pred.cpu().numpy()
    y_cpu = y.cpu().numpy()
    precision = (pred.T@y_cpu)/float(np.sum(pred))
    recall = (pred.T@y_cpu)/float(np.sum(y_cpu))
    f1 = (2*precision*recall)/(precision+recall)
    return precision, recall, f1

def func(w, X):
    return robust_sigmoid(X@w)


def write_to_txt(ws, txt):
    f = open(f"{ws}/eval_log.txt", "a")
    f.write(txt+"\n")
    print(txt)
    f.close()

    
if __name__ == "__main__":
    args = setup()
    set_seed(args.seed)
    bias = True
    ds = args.ds
    device = torch.device("cuda")
    train_X, train_y = load_data(ds=ds, split="train", bias=bias, device=device)
    test_X, test_y = load_data(ds=ds, split="test", bias=bias, device=device)
    
    for p in ["FPOR", "FROP", "OFBS"]:
        ws = f"logs/{args.ds}/{p}/{args.seed}/"
        w = np.load(f"{ws}/model.npz")['w']
        w = torch.tensor(w).to("cuda")
        
        write_to_txt(ws, f"========================= {p} ===============================")
        if p == "FPOR":
            opt_t = search_t_precision(func, w, train_X, train_y, threshold=0.8)
        elif p == "FROP":
            opt_t = search_t_recall(func, w, train_X, train_y, threshold=0.8)
        else:
            opt_t = search_t_f1(func, w, train_X, train_y)
        
        pred = (func(w, train_X)>=0.5).int()
        p, r, f1 = eval(pred, train_y)
        write_to_txt(ws, f"Train: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
        pred = (func(w, train_X)>=opt_t).int()
        p, r, f1 = eval(pred, train_y)
        write_to_txt(ws, f"\tOptimal: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")



        pred = (func(w, test_X)>=0.5).int()
        p, r, f1 = eval(pred, test_y)
        write_to_txt(ws, f"Test: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
        pred = (func(w, test_X)>=opt_t).int()
        p, r, f1 = eval(pred, test_y)
        write_to_txt(ws, f"\tOptimal: precision: {p}\t\t recall: {r}\t\t f1-score: {f1}")
