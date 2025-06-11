import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_labels
"""2类 crf"""


def dense_crf_2d(img,
                 output_probs):  # img 为H，*W*C 的原图，output_probs 为 输出概率 sigmoid 输出（h，w），#seg_map - 假设为语义分割的 mask, hxw, np.array 形式.



    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    if img.shape[-1] != 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
# """
# 测试 demo
# #image - 原始图片，hxwx3，采用 PIL.Image 读取
# #seg_map - 假设为语义分割的 mask, hxw, np.array 形式.
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# final_mask = dense_crf(np.array(image).astype(np.uint8), seg_map)
# plt.subplot(1, 3, 1)
# plt.imshow(image)
# plt.subplot(1, 3, 2)
# plt.imshow(seg_map)
# plt.subplot(1, 3, 3)
# plt.imshow(final_mask)
# plt.show()
