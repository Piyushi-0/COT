import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, SGD

import sys
sys.path.append("../")


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import pdb
import gc
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class pi_net(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(pi_net,self).__init__()
        dim1 = 1024
        dim2 = 64
        self.fc1 = nn.Linear(input_dim, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)     
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


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

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def eye_like(G):
    if(len(G.shape) == 3):
        return torch.eye(*G.shape[-2:], out=torch.empty_like(G)).repeat(G.shape[0], 1, 1)
    else: 
        return torch.eye(*G.shape,out=torch.empty_like(G))


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

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PLOT.N_CTX
        ctx_init = cfg.TRAINER.PLOT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.PLOT.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PLOT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
            nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        

        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])   
        tokenized_prompts = tokenized_prompts.repeat(self.N,1) 
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PLOT.CLASS_TOKEN_POSITION


    def forward(self):
       
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1) 
        
        ctx = ctx.permute(1, 0, 2, 3) 
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda")
        self.device1 = torch.device("cuda")
        self.N = cfg.TRAINER.PLOT.N
        self.dataset = cfg.DATASET.NAME
        self.use_uniform = True
        self.eps = 0.1
        self.max_iter = 100
        
        self.khp = cfg.khp
        self.ktype = cfg.ktype
        self.lambda_reg = cfg.lda
        
        #self.seed = cfg.seed
        
        self.lr = 1e-3
        self.wd = 1e-2
    
    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def forward(self, image, label=None):
        
        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        image_feature_pool = image_features[0]
        image_features = image_features[1:]  

        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()   
        tokenized_prompts = self.tokenized_prompts
        if self.dataset == "ImageNet":
            text_features = self.text_encoder(prompts.to(self.device1), tokenized_prompts.to(self.device1)) 
            text_features = text_features.to(self.device)
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)
        
        image_features =  F.normalize(image_features, dim=2)  # M x b x d
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)  # N x C x d
        text_feature_pool = F.normalize(text_feature_pool, dim=1)
        
        sim_op = torch.empty(b, self.n_cls, dtype=image_features.dtype, device=image_features.device)
        
        if label is not None: # training time
            classes = torch.unique(label).cpu().numpy()
            
            for _, cls in enumerate(classes):
                pinet = pi_net(50*1024, self.N).cuda()
                pinet.apply(initialize_weights)  # NOTE: Reinitializing for each batch
                pinet.train()
                
                self.opts = SGD(pinet.parameters(), lr=self.lr)
                # NOTE: not having a per-class optimizer
                cls_idx = (torch.where(label==cls)[0]).to(torch.int64)
                k = len(cls_idx)  # ideally fixed for k-shots
                kM = M*k
                
                # p(.|j^th image feature, x represented with embeddings(text+image)) over the N prompts.

                Xz_n = text_features[:, cls, :].detach()  # N x d

                Yz = image_features[:, cls_idx, :]  # M, k, d 
                
                Yz_new = Yz.permute(1, 0, 2)  # k, M, d
                # txtf = Xz_n.repeat(k, 1).contiguous().view(k, -1) # k x Nd
                # Yz_1 = (torch.cat([Yz_new.contiguous().view(k, -1), txtf], dim=1)).repeat_interleave(M, 0)  # k x Md -> kM x (M+N)d
                Yz_1 = (Yz_new.contiguous().view(k, -1)).repeat_interleave(M, 0)  # k x Md -> kM x Md
                Yz_2 = Yz_new.contiguous().view(kM, self.d)  # kM x d
                
                ZY = torch.cat([Yz_1, Yz_2], dim=1)  # kM x (Md+d) or kM x (Md+Nd+d) if text_emedding also
        
                G_xz = get_G(dist=(1-torch.einsum("md,nd->mn", Xz_n, Xz_n)) , x=Xz_n, y=Xz_n, ktype=self.ktype, khp=self.khp)
                
                v = torch.zeros(kM, dtype=image_features.dtype, device=image_features.device).fill_(1. / M)
                v1 = torch.zeros(M, dtype=image_features.dtype, device=image_features.device).fill_(1. / M)
                u = torch.zeros(self.N, dtype=image_features.dtype, device=image_features.device).fill_(1. / self.N)
                sim = torch.einsum("kmd,nd->kmn", Yz_new, Xz_n)  # k x M x N
                C = (1.0 - sim)
                
                C = F.normalize(C, p=torch.inf, dim=[1,2])
                
                
                for _ in range(self.max_iter):
                    pi_ZY = pinet(ZY)  # |YZ| x N = kM x N
                    pi = (pi_ZY*v.unsqueeze(1))
                    
                    tcost = torch.sum(pi.contiguous().view(k, M, self.N)*C)/k
                    mat = pi.sum(0).contiguous().view(self.N) - u
                    reg_x = torch.mv(G_xz, mat).dot(mat)
                    
                    obj = tcost + self.lambda_reg*reg_x/k
                    self.opts.zero_grad()
                    
                    obj.backward()
                    self.opts.step()
                    
                pinet.eval()
                for ix in range(b):
                    Yz = image_features[:, ix, :].unsqueeze(0)  # M, 1, d
                    Yz_new = Yz.permute(1, 0, 2)  # 1, M, d
                    Yz_1 = (Yz_new.contiguous().view(1, -1)).repeat_interleave(M, 0)  # 1 x Md -> M x Md
                    Yz_2 = Yz_new.contiguous().view(M, self.d)  # M x d
                    ZY = torch.cat([Yz_1, Yz_2], dim=1)
                    Yz2d = Yz_new.contiguous().view(-1, self.d)
                    pi_ZY = pinet(ZY) # M x N
                    pi = pi_ZY*v1.unsqueeze(1)  # M x N
                    Xz = text_features[:, cls, :]
                    sim = torch.einsum("md,nd->mn", Yz2d, Xz)
                    sim_op[ix, cls] = torch.sum(pi*sim)
                    
                    if label is not None:
                        torch.save(image[ix].cpu(), f"im_{ix}.pt")
                        torch.save(pi.cpu(), f"pi_{ix}.pt")
                        torch.save(label[ix].cpu(), f"label_{ix}.pt")
                    
                del pinet
                del C, G_xz, ZY, Yz, Yz_new, Yz_1, Yz_2, Yz2d, pi_ZY, pi, Xz, Xz_n, sim, mat, u, v, v1
                torch.cuda.empty_cache()
        else:
            print("Doing OT-based inference")
            sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
            sim = sim.view(M, self.N, b*self.n_cls)
            sim = sim.permute(2, 0, 1)  # torch.Size([200, 49, 4]) carries grad eurosat_2
            wdist = 1.0 - sim
            xx = torch.zeros(b*self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
            yy = torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
            # xx, yy are probabilities
            # (torch.Size([200, 49]), torch.Size([200, 4])) eurosat_2

            with torch.no_grad():
                KK = torch.exp(-wdist / self.eps)
                T = self.Sinkhorn(KK, xx, yy)  # torch.Size([200, 49, 4]) eurosat_2
            if torch.isnan(T).any():
                return None
            sim_op = torch.sum(T * sim, dim=(1, 2))
            sim_op = sim_op.contiguous().view(b, self.n_cls)
            
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature_pool @ text_feature_pool.t()
        logits2 = logit_scale * sim_op
        if self.dataset == "ImageNet":
            logits2 = (logits2 + logits)
        return logits2


@TRAINER_REGISTRY.register()
class PLOT(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PLOT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.PLOT.PREC == "fp32" or cfg.TRAINER.PLOT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()   

        print("Building custom CLIP")
        # self.pis = {}
        # for cls, _ in enumerate(classnames):
        #     self.pis[cls] = pi_net(50*1024, 4)  # dimension of Md + d x N
        #     self.pis[cls].apply(initialize_weights)
        #     self.pis[cls].to(self.device)
        #     self.pis[cls].train(True)
        self.model = CustomCLIP(cfg, classnames, clip_model) #, self.pis)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME== "ImageNet":
            self.device =  torch.device("cuda")
            # device0 = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PLOT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.PLOT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, label)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def after_epoch(self):
        # print("helloworld")
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
