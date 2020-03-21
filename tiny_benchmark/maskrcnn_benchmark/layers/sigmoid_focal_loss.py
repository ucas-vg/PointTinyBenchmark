import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _C

# TODO: Use JIT to replace CUDA implementation in the future.
class _SigmoidFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = _C.sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(
            logits, targets, d_loss, num_classes, gamma, alpha
        )
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        device = logits.device
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr


from maskrcnn_benchmark.modeling.rpn.gaussian_net.gau_label_infer import three_points_solve
class FixedIOULoss(nn.Module):
    def three_point_solve(self, li, lj, lk, a, b, eps=1e-6):
        lkj, lji = lk - lj, lj - li
        inverse_w2 = (lkj / b - lji / a) / (a + b)
        dx = -(w2 * lji / a + a) / 2
        # dx = (lkj * a * a + lji * b * b) / (lji*b - lkj * a) / 2
        return w2, dx

    def cross_points_set_solve_3d(self, L, points, a, b, step=1, solver=1):
        # points_set: (N, 3), # (c, y, x)
        """
                                 L[cj, yj-a, xj]
               L[cj, yj, xj-a]   L[cj, yj, xj]    L[cj, yj, xj + b]
                                 L[cj, yj+b, xj]
        """
        cj, yj, xj = points[:, 0], points[:, 1], points[:, 2]
        idx = torch.arange(len(points))
        lx = L[cj, yj]  # (N, W)
        lxi, lxj, lxk = lx[idx, xj - a], lx[idx, xj], lx[idx, xj + b]
        ly = L[cj, :, xj]  # (N, H) not (H, N)
        lyi, lyj, lyk = ly[idx, yj - a], lxj, ly[idx, yj + b]

        li = torch.cat([lxi, lyi], dim=0)
        lj = torch.cat([lxj, lyj], dim=0)
        lk = torch.cat([lxk, lyk], dim=0)
        s, d = self.three_point_solve(li, lj, lk, a, b)
        n = len(s) // 2
        w, h = s[:n], s[n:]
        dx, dy = d[:n], d[n:]
        # cx = xj.float() + dx  # 1/2 cause use center point
        # cy = yj.float() + dy

        # x1 = cx - (w-1/step) / 2     # notice here
        # y1 = cy - (h-1/step) / 2
        # return torch.stack([x1 * step, y1 * step, w * step, h * step, lxj], dim=1)  # lxj == lyj
        return dx, dy, w, h

    def forward(self, bbox, target, sf=0.125):
        def center2corner(dx, dy, w, h):
            l = w / 2 - dx
            r = w / 2 + dx
            t = h / 2 - dy
            b = h / 2 + dy
            return l, t, r, b
        pred_l, pred_t, pred_r, pred_b = center2corner(*bbox)
        targ_l, targ_t, targ_r, targ_b = center2corner(*target)

        l_range = (0, 4)
        pred_l = pred_l.clamp(*l_range)
        pred_r = pred_r.clamp(*l_range)
        pred_t = pred_t.clamp(*l_range)
        pred_b = pred_b.clamp(*l_range)

        target_aera = target[2] * target[3]
        pred_aera = (pred_l + pred_r) * (pred_t + pred_b)
        w_intersect = torch.min(pred_l, targ_l) + torch.min(pred_r, targ_r)
        h_intersect = torch.min(pred_b, targ_b) + torch.min(pred_t, targ_t)
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        # iou_losses = -torch.log((area_intersect.clamp(0) + 1.0) / (area_union.clamp(0) + 1.0))
        iou_losses = -torch.log(((area_intersect.clamp(0) + 1.0) / (area_union.clamp(0) + 1.0)).clamp(0.1))

        # if iou_losses.max() > 10:
        #     print("ok")
        # targ_w, targ_h = target[2], target[3]
        # l1_losses = 0.
        # for p, t, s in zip([pred_l, pred_t, pred_r, pred_b],
        #                    [targ_l, targ_t, targ_r, targ_b],
        #                    [targ_w, targ_h, targ_w, targ_h]):
        #     l1_losses += torch.log(1 + 3 * smooth_l1((p - t) / s))
        # l1_losses /= 4  # cause loss from 4 sub-loss: l, t, r, b

        # valid = ((bbox[2] > 0) & (bbox[3] > 0) & (pred_l > 0) & (pred_r > 0) & (pred_t > 0) & (pred_b > 0)).float()
        # assert (targ_h <= 0).sum() == 0 and (targ_w <= 0).sum() == 0 and (targ_l <= 0).sum() == 0 and (targ_r <= 0).sum() == 0 \
        #        and (targ_t <= 0).sum() == 0 and (targ_b <= 0).sum() == 0, ""
        # return iou_losses * valid, l1_losses * (1 - valid)
        return iou_losses * 0, iou_losses * 0


def smooth_l1(error, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(error)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


class FixSigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, sigma, fpn_strides, c, EPS=1e-6):
        super(FixSigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.EPS = EPS
        self.fpn_strides = fpn_strides
        self.c = c  # (0.5, 2, 1, 2)
        print("c1, c2, c3, c4 for pos loss:", self.c)
        self.g_mul_p = False
        self.iou_loss = FixedIOULoss()

    def forward(self, cls_logits, gau_logits, targets, valid=None):
        """
        :param logits: shape=(B, H, W, C)
        :param targets: shape=(B, H, W, C)
        :return:
        """
        gamma = self.gamma
        alpha = self.alpha
        eps = self.EPS
        c1, c2, c3, c4, c5 = self.c
        # num_classes = logits.shape[1]
        # dtype = targets.dtype
        # device = targets.device
        # # class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

        q = targets
        p = torch.sigmoid(cls_logits)
        g = torch.sigmoid(gau_logits)
        # if self.g_mul_p: g = g * p
        # loss = -(q - p) ** gamma * (torch.log(p) * alpha + torch.log(1-p) * (1 - alpha))                   # origin
        # loss = -(q - p) ** gamma * (q * torch.log(p) * alpha + (1 - q) * torch.log(1-p) * (1 - alpha))     # correct 1
        # loss = -(q - p) ** gamma * (q * torch.log(p/(q+eps)) * alpha + (1 - q) * torch.log((1-p)/(1-q+eps)) * (1 - alpha))  # correct 2

        # correct 3
        # loss = -(q - p) ** gamma * (q * torch.log(p/(q+eps)) + (1 - q) * torch.log((1-p)/(1-q+eps)))
        # neg_loss = (1-alpha) * (q <= eps).float() * loss
        # pos_loss = alpha * (q > eps).float() * loss

        # correct 4
        # loss = -(q - p) ** gamma * (q * torch.log(p/(q+eps)) + (1 - q) * torch.log((1-p)/(1-q+eps)))
        # neg_loss = (1-alpha) * (q <= eps).float() * loss
        # pos_loss = 4 * alpha * (q > eps).float() * loss

        # correct 5
        # loss = - (q * torch.log(p) + (1 - q) * torch.log(1-p))     # correct 1-2
        # neg_loss = (q <= eps).float() * (- torch.log(1 - p)) * (1 - alpha) * ((q - p) ** gamma)
        # q * |log(p) - log(q)|^2, cause inference need -log(p), so use log L2 Loss, q to weight like centerness.
        # pos_loss = q * (torch.log(p / (q + eps)) ** 2) * alpha  # * (q > eps).float()

        # loss 1
        # loss = (- q * torch.log(p) - (1 - q) * torch.log(1 - p)) * ((q - p) ** gamma)
        # neg_loss = (q <= eps).float() * loss * (1 - alpha)
        # pos_loss = (q > eps).float() * loss * alpha

        # loss 1, FL
        # loss = (- q * torch.log(p / (q + eps)) - (1 - q) * torch.log((1 - p)/(1 - q + eps))) * ((q - p) ** gamma)
        # neg_loss = (q <= eps).float() * loss * (1 - alpha)
        # pos_loss = (q > eps).float() * loss * alpha
        # print((q > eps).sum(), (q <= eps).sum())

        # # loss 2, log loss
        # neg_loss = (q <= eps).float() * (- torch.log(1 - p) * (p ** gamma)) * (1 - alpha)  # FL
        # pos_loss = (q * smooth_l1(torch.log(p / (q + eps)))) * alpha  # smoothl1([ln(p) - ln(q)])  # should be (p + eps) / (q+ eps)

        # # loss3, log diff loss
        # # use p
        # neg_loss = (q <= eps).float() * (1 - alpha) * (- p ** gamma * torch.log(1 - p))
        # pos_loss = (q > eps).float() * alpha * (- (1 - p) ** gamma * torch.log(p))
        #
        # # use g
        # gau_neg_loss = (q <= eps).float() * (1 - alpha) * (- g ** gamma * torch.log(1 - g)) * c5
        # fpn_stride, object_range, out_factor = self.fpn_strides[0], torch.Tensor([32, 64]), 2  # out_factor==2 means accept object range is [min/2, max*2]
        # # object_range[1] *= out_factor
        # # object_range[0] /= out_factor
        # # w**2=2/L * s**2(fpn_stride) in [32**2, 64**2], L in [2*(s/32)**2, 2*(s/64)**2], L*sf=[0.5, 2]
        # sf = object_range[0] / fpn_stride * object_range[1] / fpn_stride / 2  # 1/2 * (O1 * O2) / S**2=16, make 1/d2(log_q) to (0.5, 2)
        # factor = self.sigma * self.sigma * sf  # 1/diff2(log_q) in (8, 32), log_q*16 make it in (0.5, 2)
        #
        # log_p = -torch.log(g + eps) * factor
        # log_q = -torch.log(q + eps) * factor
        # center_log_p, center_log_q = log_p[:, 1:-1, 1:-1, :], log_q[:, 1:-1, 1:-1, :]
        # # qx_diff1, qy_diff1 = (center_log_q - log_q[:, :-2, 1:-1, :]), (center_log_q - log_q[:, 1:-1, :-2, :])
        # # px_diff1, py_diff1 = (center_log_p - log_p[:, :-2, 1:-1, :]), (center_log_p - log_p[:, 1:-1, :-2, :])
        # left, right = lambda x: x[:, 1:-1, :-2, :], lambda x: x[:, 1:-1, 2:, :]
        # top, bottom = lambda x: x[:, :-2, 1:-1, :], lambda x: x[:, 2:, 1:-1, :]
        # qx_diff1 = center_log_q - left(log_q)
        # qy_diff1 = center_log_q - top(log_q)
        # px_diff1 = center_log_p - left(log_p)
        # py_diff1 = center_log_p - top(log_p)
        # qx_diff2 = left(log_q) + right(log_q) - 2 * center_log_q
        # qy_diff2 = top(log_q) + bottom(log_q) - 2 * center_log_q
        # px_diff2 = left(log_p) + right(log_p) - 2 * center_log_p
        # py_diff2 = top(log_p) + bottom(log_p) - 2 * center_log_p
        # # print('px_diff', px_diff1.max(), px_diff1[qx_diff1 > 0].mean())
        # # print('qy_diff', qy_diff1.max(), qy_diff1[qy_diff1 > 0].mean())
        # # valid_x = (q[:, :-2, 1:-1, :] > eps) & (q[:, 2:, 1:-1, :] > eps)
        # # valid_y = (q[:, 1:-1, :-2, :] > eps) & (q[:, 1:-1, 2:, :] > eps)
        #
        # # abs(dx) = s/8/2, (32, 64) -> t in (2, 4), (-tf/2, tf/2)
        # tf = (object_range[1] / fpn_stride)
        # dqx = -((qx_diff1+eps) / (qx_diff2+eps) + 0.5)[valid] / tf
        # dqy = -((qy_diff1+eps) / (qy_diff2+eps) + 0.5)[valid] / tf
        # dpx = -((px_diff1+eps) / (qx_diff2+eps) + 0.5)[valid] / tf  # use qx_diff2, not px_diff2 to get smooth grad.
        # dpy = -((py_diff1+eps) / (qy_diff2+eps) + 0.5)[valid] / tf
        # x_loss = torch.log(1 + 3 * (dqx - dpx).clamp(-1, 1).abs())
        # y_loss = torch.log(1 + 3 * (dqy - dpy).clamp(-1, 1).abs())
        # xy_loss = (smooth_l1(x_loss, beta=0.25) + smooth_l1(y_loss, beta=0.25))
        #
        # d2_range = 1./2/out_factor, 2 * out_factor
        # px_diff2 = px_diff2.clamp(*d2_range)[valid]
        # py_diff2 = py_diff2.clamp(*d2_range)[valid]
        # qx_diff2 = qx_diff2.clamp(*d2_range)[valid]
        # qy_diff2 = qy_diff2.clamp(*d2_range)[valid]
        #
        # gau_loss = (q[:, 1:-1, 1:-1, :] > 0).float() * smooth_l1(center_log_p - center_log_q)
        # wh_loss = (smooth_l1(c3 * torch.log(qx_diff2/px_diff2), beta=0.25) +
        #            smooth_l1(c3 * torch.log(qy_diff2/py_diff2), beta=0.25))
        #
        # # def ri(x): return round(x.item(), 3)
        # # print("neg_loss", ri(neg_loss.max()), ri(neg_loss.mean()), end=';')
        # #
        # # def ri(x): return round(x.item(), 3) if valid.sum() > 0 else 0
        # # print('gau_loss', ri(gau_loss.max()), ri(gau_loss.mean()), end=";")
        # # print('wh_loss', ri(wh_loss.max()), ri(wh_loss.mean()), end=';')
        # # print('xy_loss', ri(xy_loss.max()), ri(xy_loss.mean()), )
        # valid_q = q[:, 1:-1, 1:-1, :][valid]
        # gau_loss = q[:, 1:-1, 1:-1, :] * (c1*gau_loss)
        # wh_loss = valid_q * (c2*wh_loss)
        # xy_loss = valid_q * (c4*xy_loss)
        # return neg_loss.sum(), pos_loss.sum(), gau_neg_loss.sum() * 0, gau_loss.sum(), wh_loss.sum(), xy_loss.sum()

        # loss4, IOU
        neg_loss = (q <= eps).float() * (1 - alpha) * (- p ** gamma * torch.log(1 - p))
        pos_loss = (q > eps).float() * alpha * (- (1 - p) ** gamma * torch.log(p))

        g = g.permute((0, 3, 1, 2))
        q = q.permute((0, 3, 1, 2))
        valid = valid.permute((0, 3, 1, 2))

        factor = self.sigma * self.sigma
        log_p = -torch.log(g + eps) * factor
        log_q = -torch.log(q + eps) * factor

        fpn_stride, object_range, out_factor = self.fpn_strides[0], torch.Tensor([32, 64]), 2
        sf = 1 / ((object_range[0] / fpn_stride * object_range[1] / fpn_stride) ** 0.5)

        iou_losses = 0.
        l1_losses = 0.
        for b in range(len(valid)):
            idx = torch.nonzero(valid[b])
            if len(idx) == 0: continue
            idx[:, 1:] += 1
            p_bboxes = self.iou_loss.cross_points_set_solve_3d(log_p[b], idx, 1, 1, step=1, solver=1)
            q_bboxes = self.iou_loss.cross_points_set_solve_3d(log_q[b], idx, 1, 1, step=1, solver=1)
            iou_loss, l1_loss = self.iou_loss(p_bboxes, q_bboxes, sf)
            valid_q = q[b, :, 1:-1, 1:-1][valid[b]]
            iou_losses += (valid_q * iou_loss).sum()
            l1_losses += (valid_q * l1_loss).sum()

        def ri(x): return round(x.item(), 3)
        print("neg_loss", ri(neg_loss.max()), ri(neg_loss.mean()), end=';')
        print(iou_losses, l1_losses)
        return neg_loss.sum(), pos_loss.sum(), iou_losses * 0, l1_losses * 0


class L2LossWithLogit(nn.Module):
    def __init__(self):
        super(L2LossWithLogit, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        return self.mse(p, targets)
