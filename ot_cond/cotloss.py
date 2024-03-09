import torch
from ot_cond.utils import set_seed, get_preds


def train_pi_v2(to_train, train_dataloader, optimizer, scheduler, f, pi_net, C, G, lambda_reg, reg_type="mmd_sq", bsize=None, d=None, SEED=None, y_as_labels=1):
    set_seed(SEED)
    if to_train == "pi":
        for params in pi_net.parameters():
            params.requires_grad = True
        pi_net.train()
        for params in f.parameters():
            params.requires_grad = False
        f.eval()
    else:
        for params in pi_net.parameters():
            params.requires_grad = False
        pi_net.eval()
        for params in f.parameters():
            params.requires_grad = True
        f.train()

    cmf = C.flatten()
    n_cls = C.shape[0]
    device = C.device
    dtype = C.dtype
    onehot_mat = torch.eye(n_cls).to(dtype).to(device)
    
    size = len(train_dataloader.dataset)
    loss = 0.
    tot_transport_loss = 0.
    tot_reg_cost = 0.
    
    if bsize is not None:
        ZX = torch.empty(n_cls*bsize, d+n_cls).to(dtype).to(device)  # predeclaring to avoid memory leak
        X_mat = torch.empty(n_cls*bsize).to(dtype).to(device)  # predeclaring to avoid memory leak
    
    for _, (Z, y) in enumerate(train_dataloader):
        bsize = len(Z)
        X_mat = onehot_mat.repeat(bsize, 1)
        Z, y = Z.to(device), y.to(device)
        ZX = torch.cat([Z.repeat_interleave(n_cls, 0), X_mat], dim=1)
        pi_ZX = pi_net(ZX)
        f_Z = f(Z).flatten()
        
        pi = pi_ZX*f_Z.unsqueeze(1)
         
        p_qs = pi.unfold(0, n_cls, n_cls).sum(2)
        
        transport_loss = torch.tensordot(pi.reshape(-1, n_cls*n_cls), cmf.unsqueeze(0))/bsize
        tot_transport_loss += transport_loss
        
        if not y_as_labels:
            new_p_qs = p_qs - y
        else:
            delta_gqs = torch.nn.functional.one_hot(y, num_classes=n_cls).to(dtype).to(device)
            new_p_qs = p_qs - delta_gqs
        
        if reg_type == "mmd":
            reg_cost = 0
            for vec in new_p_qs:
                reg_cost = reg_cost + torch.sqrt(torch.mv(G, vec).dot(vec))
        else:
            reg_cost = torch.tensordot(torch.mm(new_p_qs, G), new_p_qs)
        tot_reg_cost += reg_cost
        
        loss = transport_loss + lambda_reg*reg_cost/bsize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    
    return pi_net, f, tot_transport_loss/size, tot_reg_cost/size, optimizer, scheduler


def eval_loss(dataloader, f, pi_net, C, G, lambda_reg, SEED=None, y_as_labels=1):
    set_seed(SEED)
    pi_net.eval()
    f.eval()
    
    cmf = C.flatten()
    n_cls = C.shape[0]
    device = C.device
    dtype = C.dtype
    onehot_mat = torch.eye(n_cls).to(dtype).to(device)
    
    size = len(dataloader.dataset)
    correct = 0
    tot_reg_cost = 0.
    tot_transport_loss = 0.
    with torch.no_grad():
        for _, (Z, y) in enumerate(dataloader):
            Z, y = Z.to(device), y.to(device)
            
            correct += (get_preds(f, Z) == y).type(dtype).sum().item()
            
            bsize = len(Z)
            X_mat = onehot_mat.repeat(bsize, 1)
            
            ZX = torch.cat([Z.repeat_interleave(n_cls, 0), X_mat], dim=1)
            pi_ZX = pi_net(ZX)
            f_Z = f(Z).flatten()
            
            pi = pi_ZX*f_Z.unsqueeze(1)
            
            tot_transport_loss += torch.tensordot(pi.reshape(-1, n_cls*n_cls), cmf.unsqueeze(0))/bsize
            
            p_qs = pi.unfold(0, n_cls, n_cls).sum(2)  #  pi.sum(1).reshape(bsize, n_cls)

            if not y_as_labels:
                new_p_qs = p_qs - y
            else:
                delta_gqs = torch.nn.functional.one_hot(y, num_classes=n_cls).to(dtype).to(device)
                new_p_qs = p_qs - delta_gqs
            
            tot_reg_cost += torch.tensordot(torch.mm(new_p_qs, G), new_p_qs)

    return tot_transport_loss/size, tot_reg_cost/size, correct/size
