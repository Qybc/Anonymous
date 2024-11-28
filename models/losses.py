import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
import numpy as np

import ot
# from sentence_transformers import util



# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True

# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()

# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def all_gather_batch(tensors):
#     """
#     Performs all_gather operation on the provided tensors.
#     """
#     # Queue the gathered tensors
#     world_size = get_world_size()
#     # There is no need for reduction in the single-proc case
#     if world_size == 1:
#         return tensors
#     tensor_list = []
#     output_tensor = []
#     for tensor in tensors:
#         tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
#         dist.all_gather(
#             tensor_all,
#             tensor,
#             async_op=False  # performance opt
#         )

#         tensor_list.append(tensor_all)

#     for tensor_all in tensor_list:
#         output_tensor.append(torch.cat(tensor_all, dim=0))
#     return output_tensor

softmax_helper = lambda x: F.softmax(x, 1)

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input, target):
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
    
def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
    
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn
    
class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        # import pdb;pdb.set_trace()

        if not self.do_bg:
            if self.batch_dice:
                try :
                    dc = dc[1:]
                except:
                    dc = dc.cpu().detach().numpy()[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc
    
class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        # import pdb;pdb.set_trace()

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result
    
class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            curr_band_width = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
            print("kernel bandwidth: ", curr_band_width)
            return curr_band_width

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(X.device))[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, bandwidth=None):
        super().__init__()
        self.kernel = RBF()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    

class MahalalobisLoss(nn.Module):

    def __init__(self, epsilon=0.05, reg=0.1, m=0.95, L=None, pot=False):
        super(MahalalobisLoss, self).__init__()
        self.epsilon = epsilon
        self.reg = reg # 0.1
        self.mmd_reg = MMDLoss()
        self.mmd_reg.cuda()
        self.m =m
        self.POT = pot

    def forward(self, audio_emb, text_emb, M): # M: 1024 1024
        batch_size = audio_emb.size(0)
        a = torch.ones(batch_size)/batch_size
        b = torch.ones(batch_size)/batch_size
        a = a.to(audio_emb.device)
        b = b.to(audio_emb.device)

        pi_hat = torch.eye(batch_size).to(audio_emb.device)/(batch_size)

        M = torch.nan_to_num(M)
        u, s, v =torch.svd(M)
        reg = torch.sum(s)

        audio_matrix = audio_emb.unsqueeze(0).repeat(audio_emb.size(0),1,1) # bs 1024 -> bs bs 1024
        text_matrix = text_emb.unsqueeze(1).repeat(1, text_emb.size(0), 1) # bs 1024 -> bs bs 1024

        pairwise_dist = audio_matrix - text_matrix # bs bs 1024
        t_pairwise_dist = pairwise_dist.transpose(1,2) # bs 1024 bs

        M_dist = torch.einsum("ijk,ikj,kk->ij", pairwise_dist, t_pairwise_dist, M)
        M_dist = torch.sqrt(M_dist) # 马氏距离的计算
        M_dist = M_dist/M_dist.max() # bs bs

        if self.POT:
            pi = ot.partial.entropic_partial_wasserstein(a,b,M_dist, reg=self.epsilon, m=self.m) # [bs,bs] 
        else:
            pi = ot.sinkhorn(a,b,M_dist, reg=self.epsilon)
        
        ot_loss = -pi_hat*torch.log(pi) # pi_hat是一个正定阵，对角都是0.0417=1/bs
        ot_loss = torch.sum(ot_loss)

        loss = ot_loss + self.reg*reg
        loss = ot_loss

        # import pdb;pdb.set_trace()

        return loss
    
class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeds, text_embeds, labels):
        """

        :param audio_embeds:
        :param text_embeds:
        :param labels:
        :return:
        """

        n = audio_embeds.size(0)  # batch size

        # dist = []
        sim_a2t = util.cos_sim(audio_embeds, text_embeds)  # (batch_size, x batch_size)
        sim_ap = torch.diag(sim_a2t).view(n, 1)
        d1 = sim_ap.expand_as(sim_a2t)
        d2 = sim_ap.t().expand_as(sim_a2t)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim_a2t - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim_a2t - d2)

        # clear diagonals
        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(cost_a.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        cost_s = cost_s.max(1)[0]
        cost_a = cost_a.max(0)[0]

        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


class NTXent(nn.Module): # contrastive loss

    def __init__(self, temperature=0.07, epsilon=0.1):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature
        self.epsilon = epsilon

    def forward(self, audio_embeds, text_embeds, labels):

        n = batch_size = audio_embeds.shape[0]

        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(a2t.device)
        mask_diag = mask.diag()
        mask_diag = torch.diag_embed(mask_diag)
        mask = mask ^ mask_diag

        a2t_loss = - self.loss(a2t).masked_fill(mask, 0).diag().mean()
        t2a_loss = - self.loss(t2a).masked_fill(mask, 0).diag().mean()
        
        prob_a2t = torch.nn.functional.softmax(a2t, dim=-1)
        ent_a2t = torch.mean(torch.sum(prob_a2t*torch.log(prob_a2t), dim=-1))
        
        prob_t2a = torch.nn.functional.softmax(t2a, dim=-1)
        ent_t2a = torch.mean(torch.sum(prob_t2a*torch.log(prob_t2a), dim=-1))

        ent_reg = self.epsilon*(ent_a2t + ent_t2a)
        # print("Entropy reg: ", ent_reg)
        # loss = 0.5 * a2t_loss + 0.5 * t2a_loss - ent_reg
        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss


class Med3DWith2DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.ot_loss = MahalalobisLoss(epsilon=0.03, reg=0, m=0.95, pot=True)
        self.res_loss = nn.MSELoss()

    def forward(self, outputs, image_embedding, text_embedding, M, logit_scale=None, align_type='optimal_transport'):

        med_embedding, med_x, med_res = outputs


        if logit_scale is None:
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        # import pdb;pdb.set_trace()

        # normalized features: bs 512
        med_embed = F.normalize(med_embedding, dim=-1, p=2)
        text_embed = F.normalize(text_embedding, dim=-1, p=2).float() # f16
        image_embed = F.normalize(image_embedding, dim=-1, p=2).float() # f16

        if align_type == 'contrastive':
            # cosine similarity as logits
            logits_per_pc_text = logit_scale * med_embed @ text_embed.t()
            logits_per_text_med = logit_scale * text_embed @ med_embed.t()
            logits_per_med_image = logit_scale * med_embed @ image_embed.t()
            logits_per_image_med = logit_scale * image_embed @ med_embed.t()

            loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                    F.cross_entropy(logits_per_text_med, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_med_image, self.labels) + F.cross_entropy(logits_per_image_med, self.labels)) / 2
        if align_type == 'optimal_transport':
            # import pdb;pdb.set_trace()
            tcl_loss = self.ot_loss(med_embed, image_embed, M) + self.ot_loss(med_embed, text_embed, M)
        if align_type == 'triplet':
            loss = 0

        res_loss = self.res_loss(med_x, med_res)
        loss = tcl_loss + res_loss
        return loss