from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib import pyplot as plt
import torch
import logging
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import torch.nn as nn

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def createLogHandler(log_file, job_name="_"):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_file, mode='a')
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s; , %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def eye_like(G):
    if(len(G.shape) == 3):
        return torch.eye(*G.shape[-2:], out=torch.empty_like(G)).repeat(G.shape[0],1,1)
    else: 
        return torch.eye(*G.size(),out=torch.empty_like(G))

def set_seed(SEED=0):
    if SEED is None:
        return
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def get_preds(f, inp_batch):
    preds = f(inp_batch).argmax(dim=1)
    return preds

def plot_decision_b(net, z, y, title="Decision Boundary"):
    feature_1, feature_2 = torch.meshgrid(torch.linspace(z[:, 0].min().item(), z[:, 0].max().item(), 50),
                                          torch.linspace(z[:, 1].min().item(), z[:, 1].max().item(), 50))
    grid = (torch.vstack([feature_1.ravel(), feature_2.ravel()]).T).to(z.dtype).to(z.device)
    
    y_preds = get_preds(net, grid).view(feature_1.shape)
    z_0 = z[:, 0].cpu().numpy()
    z_1 = z[:, 1].cpu().numpy()
    y = y.cpu().numpy()
    
    display = DecisionBoundaryDisplay(xx0=feature_1.cpu().numpy(), xx1=feature_2.cpu().numpy(), response=y_preds.cpu().numpy())
    display.plot()
    display.ax_.scatter(z_0, z_1, c=y, edgecolors="black")
    plt.title(title)
    plt.show()

def test_conv(obj_itr, criteria="obj"):
    if criteria == "obj":
        prv_obj = obj_itr[-2]
        cur_obj = obj_itr[-1]
        rel_dec = abs(prv_obj-cur_obj)/(abs(prv_obj)+1e-10)
        if rel_dec < 1e-6:
            return 1
        return 0
    if criteria == "rgrad":
        raise NotImplementedError
    
def get_dist(x, y, p=2, dtype="euc", khp=None):
    x = x.unsqueeze(1) if x.dim() == 1 else x
    y = y.unsqueeze(1) if y.dim() == 1 else y

    C = torch.cdist(x, y)

    if p == 2 or "ker" in dtype:
        C = C**2
        if "rbf" in dtype:
            C = 2-2*get_G(dist=C, ktype="rbf", khp=khp, x=x, y=y)
        if "imq" in dtype:
            C = 2/khp**(0.5)-2*get_G(dist=C, ktype="imq", khp=khp, x=x, y=y)
    if "ker" in dtype and p == 1:
        C = C**(0.5)
    return C

def get_G(dist=None, ktype="rbf", khp=None, x=None, y=None, ridge=1e-10):
    """
    # NOTE: if dist is not None, it should be cost matrix**2. 
    If it is None, the function automatically computes euclidean**2.
    """
    if ktype in ["rbf", "imq"]:
        if khp == None:  # take median heuristic
            khp = 0.5*torch.median(get_dist(x, y, "p1").view(-1))
        if dist is None:
            dist = get_dist(x, y)
    if ktype == "lin":
        G = torch.einsum('md,nd->mn', x, y)
    elif ktype == "rbf":
        G = torch.exp(-dist/(2*khp))
    elif ktype == "imq":
        G = (khp + dist)**(-0.5)
    if(len(G.shape)==2):
        if G.shape[0] == G.shape[1]:
            G = (G + G.T)/2
    elif(G.shape[1] == G.shape[2]):
        G = (G + G.permute(0,2,1))/2
    G = G + ridge*eye_like(G)
    return G


def get_G_v2(dist=None, ktype="rbf", khp=None, x=None, y=None, ridge=1e-10):
    """
    Gram Matrix?
    # NOTE: if dist is not None, it should be cost matrix**2. 
    If it is None, the function automatically computes euclidean**2.
    """
    if ktype in ["rbf", "imq", "imq_v2"]:
        if khp == None or khp == -1:  # take median heuristic
            khp = 0.5*torch.median(get_dist(x, y).view(-1))
        if dist is None:
            dist = get_dist(x, y)
    if ktype == "lin":
        if x.dim() == 2:
            G = torch.einsum('md,nd->mn', x, y)
        else:
            G = torch.einsum('bmd,nd->bmn', x, y)
    elif ktype == "rbf":
        G = torch.exp(-dist/(2*khp))
    elif ktype == "imq":
        G = (khp + dist)**(-0.5)
    elif ktype == "imq_v2":
        G = ((1+dist)/khp)**(-0.5)

    if(len(G.shape)==2):
        if G.shape[0] == G.shape[1]:
            G = (G + G.T)/2
    elif(G.shape[1] == G.shape[2]):
        G = (G + G.permute(0, 2, 1))/2
    G = G + ridge*eye_like(G)
    return G

def get_G_v3(dist=None, ktype="rbf", khp=None, x=None, y=None, ridge=1e-10):
    """
    Same as get_G_v2 but includes median heuristics for a batch of inputs.
    """
    if ktype not in ["lin"]:
        if dist is None:
            dist = get_dist(x, y)
        if khp == None or khp==-1:  # take median heuristic
            b = len(dist.shape)
            if b==2:
                khp = 0.5*torch.median(get_dist(x, y, "p1").view(-1))
            else:
                khp = 0.5*torch.median(get_dist(x, y, p=1).view(dist.shape[0], -1), dim=1)[0]
                khp = khp.view(-1, 1, 1)
            
    if ktype == "lin":
        if x.dim() == 2:
            G = torch.einsum('md,nd->mn', x, y)
        else:
            G = torch.einsum('bmd,nd->bmn', x, y)
    elif ktype == "rbf":
        G = torch.exp(-dist/(2*khp))
    elif ktype == "imq":
        G = (khp + dist)**(-0.5)
    elif ktype == "imq_v2":
        G = ((1+dist)/khp)**(-0.5)
    if(len(G.shape)==2):
        if G.shape[0] == G.shape[1]:
            G = (G + G.T)/2
    elif(G.shape[1] == G.shape[2]):
        G = (G + G.permute(0,2,1))/2
    G = G + ridge*eye_like(G)
    return G

def offd_dist(M, minn):
    """For deciding sigma based on distance statistics."""
    import heapq
    def next_nsmallest(numbers, n):
        nsmallest = {}
        for i in range(1, n):
            nsmallest[i] = heapq.nsmallest(i+1, numbers)[-1]
        return nsmallest
    ofdM = M[~torch.eye(M.shape[0], dtype=bool)]
    min_dist = torch.min(ofdM)
    max_dist = torch.max(ofdM)
    med_dist = torch.median(ofdM)
    minn_dist = next_nsmallest(torch.unique(ofdM).cpu().numpy(), minn)

    return {"min_dist": min_dist, "max_dist": max_dist, "med_dist": med_dist,
            "minn_dist": minn_dist}

def get_auc_acc(dataloader, f, device, dtype, with_softmax=1, y_as_labels=0):
    f.eval()
    f.to(device)
    correct = 0
    
    size = len(dataloader.dataset)
    all_gt = []
    all_pred = []
    with torch.no_grad():
        for Z, y in dataloader:
            Z, y = Z.to(device), y.to(device)
            
            if with_softmax:
                pred_softmax = f(Z)
            else:
                pred_softmax = nn.functional.softmax(f(Z), dim=1)
            pred = pred_softmax.argmax(dim=1)
            
            y_labels = y if y_as_labels else torch.where(y)[1]
            correct += (pred == y_labels).to(dtype).sum().item()
            
            y_gt_labels = y if not y_as_labels else \
            torch.nn.functional.one_hot(y.to(torch.long) , num_classes=pred_softmax.shape[1]).to(dtype).to(device)
            all_gt.extend(y_gt_labels.cpu())
            all_pred.extend(pred_softmax.cpu())
            
    all_gt = torch.stack(all_gt).numpy()
    all_pred = torch.stack(all_pred).numpy()
    auc = roc_auc_score(all_gt, all_pred)
    return correct/size, auc

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

