import torch
import torch.nn as nn
import math
from sklearn.utils.extmath import cartesian
import numpy as np

from ..builder import LOSSES


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||

    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances


def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    return res


@LOSSES.register_module()
class WeightedHausdorffDistance(nn.Module):
    def __init__(self, p=-9, return_2_terms=False):
        """
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        """
        super(WeightedHausdorffDistance, self).__init__()

        self.height, self.width = None, None
        self.resized_size = None
        self.max_dist = None
        self.n_pixels = None
        self.all_img_locations = None

        self.return_2_terms = return_2_terms
        self.p = p

    def point_generator(self, resized_height, resized_width, device):
        """
        Args:
            :param resized_height: Number of rows in the image.
            :param resized_width: Number of columns in the image.
            :param device: Device where all Tensors will reside.
        Returns:
            all_img_locations: Tensor, shape is [n_pixels, 2], all points coord (x, y) in images
        """
        if self.height is None and self.width is None:
            # Prepare all possible (row, col) locations in the image
            self.height, self.width = resized_height, resized_width
            self.resized_size = torch.tensor([resized_height, resized_width],
                                             dtype=torch.get_default_dtype(), device=device)
            self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
            self.n_pixels = resized_height * resized_width
            self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                                 np.arange(resized_width)]))
            # Convert to appropiate type
            self.all_img_locations = self.all_img_locations.to(device=device,
                                                               dtype=torch.get_default_dtype())
        else:
            assert self.height == resized_height and self.width == resized_width, "image size should be fixed."

    def forward(self, prob_map, gt):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """
        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size, resize_height, resize_width = prob_map.shape
        assert batch_size == len(gt)
        self.point_generator(resize_height, resize_width, prob_map.device)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            # orig_size_b = orig_sizes[b, :]
            # norm_factor = (orig_size_b / self.resized_size).unsqueeze(0)

            # Corner case: no GT points
            if (gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0) or len(gt_b) == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            d_matrix = cdist(self.all_img_locations, gt_b)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * \
                torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated) * self.max_dist + p_replicated * d_matrix
            minn = generaliz_mean(weighted_d_matrix, p=self.p, dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        return terms_1.mean(), terms_2.mean()

