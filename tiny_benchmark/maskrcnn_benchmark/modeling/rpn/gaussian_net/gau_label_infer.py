import torch
import matplotlib.pyplot as plt
import numpy as np


def generate_score_map(image_size, boxes, beta, sigma, step=1):
    # boxes: (N, 4), (x1 y1, w, h)
    h, w = image_size

    X = torch.arange(0, w).float()
    Y = torch.arange(0, h).float()
    Y, X = torch.meshgrid(Y, X)   # X, Y shape is (len(arg1), len(arg2), ...len(arg_n))

    X, Y = X.reshape((-1, )), Y.reshape((-1,))

    x1, y1, W, H = boxes.transpose(0, 1)
    cx = x1 + (W-1) / 2
    cy = y1 + (H-1) / 2
    D = ((X[:, None] - cx[None]) / (sigma * W[None, :])) ** beta + \
        ((Y[:, None] - cy[None]) / (sigma * H[None, :])) ** beta
    Q = torch.exp(-D)

    label = Q.max(dim=1)[0]
    score_map = label.reshape((1, 1, h, w))

    return score_map[:, :, ::step, :: step]


def add_noise(I, std=0.1):
    noise = torch.randn(I.shape) * std
    return I + noise


def add_boxes(ax, boxes):
    x1, y1, W, H = boxes.transpose(0, 1).numpy()
    for i in range(len(boxes)):
        ax.add_patch(plt.Rectangle((x1[i], y1[i]), W[i], H[i], fill=False))


def three_points_solve1(li, lj, lk, a, b, eps=1e-6):
    """
        (1) + (2)
    """
    lkj, lji = lk - lj, lj - li
    w2 = (a + b) / (lkj / b - lji / a + eps)
    dx = -(w2 * lji / a + a) / 2
    # dx = (lkj * a * a + lji * b * b) / (lji*b - lkj * a) / 2
    return w2, dx


def three_points_solve2(li, lj, lk, a, b, eps=1e-6):
    lki, lji = lk - li, lj - li
    w2 = b / (lki / (a + b) - lji / a + eps)
    dx = -(w2 * (lji / a) + a) / 2
    # dx = (lki * a * a + lji * (b*b -a*a)) / (lji*(a + b) - lki * a) / 2
    return w2, dx


def three_points_solve3(li, lj, lk, a, b, eps=1e-6):
    lkj, lki = lk - lj, lk - li
    w2 = a / (lkj / b - lki / (a + b) + eps)
    dx = (b - w2 * lkj / b) / 2
    # dx = (lkj * (a+b) * (a+b) - lki * b * b) / (lkj * (a + b) - lki * b) / 2
    return w2, dx


def three_points_solve(li, lj, lk, a, b, return_w2=False, eps=1e-6, solver=1):
    if solver == 1:
        three_points_solve_ = three_points_solve1
    elif solver == 2:
        three_points_solve_ = three_points_solve2
    elif solver == 3:
        three_points_solve_ = three_points_solve3
    else:
        raise ValueError("solver only support 1, 2, 3")

    w2, dx = three_points_solve_(li, lj, lk, a, b, eps)
    if return_w2:
        return w2, dx
    else:
        w2 = w2.clamp(0)
        w = w2 ** 0.5
        return w, dx


def cross_points_set_solve_2d(L, points, a, b, step=1):
    # points_set: (N, 2)
    """
                        L[yj-a, xj]
           L[yj, xj-a]  L[yj, xj]   L[yj, xj + b]
                        L[yj+b, xj]
    """
    xj, yj = points[:, 0], points[:, 1]
    idx = torch.arange(len(points))
    lx = L[yj]
    lxi, lxj, lxk = lx[idx, xj - a], lx[idx, xj], lx[idx, xj + b]
    ly = L[:, xj]
    lyi, lyj, lyk = ly[yj - a, idx], lxj, ly[yj + b, idx]

    li = torch.cat([lxi, lyi], dim=0)
    lj = torch.cat([lxj, lyj], dim=0)
    lk = torch.cat([lxk, lyk], dim=0)
    s, d = three_points_solve(li, lj, lk, a, b)
    n = len(s) // 2
    w, h = s[:n], s[n:]
    dx, dy = d[:n], d[n:]
    cx = xj.float() + dx
    cy = yj.float() + dy

    # x1 + x2 = 2*x1 + w - 1 = 2*cx
    # cx = x1 + (w-1)/2
    x1 = cx - (w-1/step) / 2     # notice here
    y1 = cy - (h-1/step) / 2
    return torch.stack([x1 * step, y1 * step, w * step, h * step, lxj], dim=1)  # lxj == lyj


def cross_points_set_solve_3d(L, points, a, b, step=1, solver=1):
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
    s, d = three_points_solve(li, lj, lk, a, b, solver=solver)
    n = len(s) // 2
    w, h = s[:n], s[n:]
    dx, dy = d[:n], d[n:]
    cx = xj.float() + dx  # 1/2 cause use center point
    cy = yj.float() + dy

    # x1 + x2 = 2*x1 + w - 1 = 2*cx
    # cx = x1 + (w-1)/2
    x1 = cx - (w-1/step) / 2     # notice here
    y1 = cy - (h-1/step) / 2
    return torch.stack([x1 * step, y1 * step, w * step, h * step, lxj], dim=1)  # lxj == lyj
    # return torch.stack([x1, y1, w, h, lxj], dim=1)  # lxj == lyj


def xcyc2x1y1(xc, yc, w, h):
    x1 = xc - (w - 1) / 2
    y1 = yc - (h - 1) / 2
    return x1, y1, w, h


if __name__ == "__main__":
    # set beta and sigma for gaussian
    beta = 2
    inflection_point = 0.25
    sigma = inflection_point * ((beta / (beta - 1)) ** (1.0 / beta))

    # # generate fake score map
    # gts = torch.Tensor([[200, 300, 101, 91]]).float()     # boxes: (N, 4), (x1 y1, w, h)
    # S = generate_score_map((800, 1333), gts, beta, sigma)
    #
    # # show score map
    # I = S[0, 0]
    # plt.imshow(I)
    # add_boxes(plt.axes(), gts)
    # plt.show()
    #
    # # test three_points_solve by give one point and a, b
    # L = -torch.log(I) * sigma * sigma
    # (x, y), a, b = (200, 300), 1, 2
    # w, dx = three_points_solve(L[y, x-a], L[y, x], L[y, x+b], a, b)
    # h, dy = three_points_solve(L[y-a, x], L[y, x], L[y+b, x], a, b)
    # print("(cx, cy), w, h:", (x + dx, y + dy), w, h)
    # print(L)
    #
    # # # test cross_points_set_solve with 30 random points
    # # x1, y1, K = gts[:, 0], gts[:, 1], 30
    # # points = (torch.rand(K, 2) * 90).long()
    # # i = (torch.rand(K) * len(gts)).long()
    # # points[:, 0] += x1[i].long()
    # # points[:, 1] += y1[i].long()
    # # boxes = cross_points_set_solve_2d(L, points, 1, 1)
    # #
    # # gts_score = torch.cat([gts, torch.Tensor([[0]])], dim=1)
    # # print(boxes)
    # # # print(torch.Tensor(sorted((boxes - gts_score[0]).abs().numpy().tolist(), key=lambda x: -x[-1])))
    #
    # I_hat = add_noise(I)
    # L = -torch.log(I_hat) * sigma * sigma
    # # test cross_points_set_solve with 30 random points
    # x1, y1, K = gts[:, 0], gts[:, 1], 30
    # points = (torch.rand(K, 2) * 90).long()
    # i = (torch.rand(K) * len(gts)).long()
    # points[:, 0] += x1[i].long()
    # points[:, 1] += y1[i].long()
    # boxes = cross_points_set_solve(L, points, 1, 1)
    #
    # gts_score = torch.cat([gts, torch.Tensor([[0]])], dim=1)
    # print(boxes)

    step = 4
    # generate fake score map
    gts1 = torch.Tensor([[200, 300, 101, 91]]).float()     # boxes: (N, 4), (x1 y1, w, h)
    S1 = generate_score_map((800, 1333), gts1, beta, sigma, step=step)
    gts2 = torch.Tensor([[300, 200, 51, 67]]).float()
    S2 = generate_score_map((800, 1333), gts2, beta, sigma, step=step)
    S = torch.cat([S1, S2], dim=1)
    print(S.shape)
    L = -torch.log(S) * sigma * sigma

    # (x, y), a, b = (200, 300), 1, 1
    # w, dx = three_points_solve(L[0, 0, y, x-a], L[0, 0, y, x], L[0, 0, y, x+b], a, b)
    # h, dy = three_points_solve(L[0, 0, y-a, x], L[0, 0, y, x], L[0, 0, y+b, x], a, b)
    # print("x1, y1, w, h:", xcyc2x1y1(x + dx, y + dy, w, h))
    gts = torch.cat([gts1, gts2], dim=0)
    x1, y1, K = gts[:, 0], gts[:, 1], 30
    points = (torch.rand(K, 2) * 50).long()
    points = torch.cat([(torch.rand(K, 1) * 2).long(), points], dim=1)
    i = (torch.rand(K) * len(gts)).long()
    points[:, 1] += y1[points[:, 0]].long()
    points[:, 2] += x1[points[:, 0]].long()
    points[:, [1, 2]] /= step
    boxes = cross_points_set_solve_3d(L[0], points, 1, 1, step=step)
    print(torch.cat([
        boxes[:, :4] - gts[points[:, 0]],
        boxes[:, -1:]
    ], dim=1))
