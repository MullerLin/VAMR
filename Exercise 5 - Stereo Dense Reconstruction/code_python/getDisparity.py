import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def getDisparity(left_img, right_img, patch_radius, min_disp, max_disp):
    """
    left_img and right_img are both H x W and you should return a H x W matrix containing the disparity d for
    each pixel of left_img. Set disp_img to 0 for pixels where the SSD and/or d is not defined, and for
    d estimates rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy
    min_disp <= d <= max_disp.
    """
    r_p = patch_radius
    patch_size = 2 * r_p + 1
    img_dis = np.zeros(left_img.shape).astype(np.float)
    n_row, n_col = left_img.shape
    for row in range(r_p, n_row - r_p):
        for col in range(max_disp + r_p, n_col - r_p):  # p1 = p0 -d
            left_patch = left_img[(row - r_p):(row + r_p + 1), (col - r_p):(col + r_p + 1)]
            right_strip = right_img[(row - r_p):(row + r_p + 1), (col - r_p - max_disp):(col + r_p - min_disp + 1)]

            right_patches = np.zeros([patch_size, patch_size, max_disp - min_disp + 1])
            for i in range(0, max_disp - min_disp + 1):
                right_patches[:, :, i] = right_strip[:, i:i + 2 * r_p + 1]

            leftP_vec = left_patch.flatten()
            rightPs_vecs = right_patches.reshape([patch_size ** 2, max_disp - min_disp + 1])
            ssds = cdist(leftP_vec[None, :], rightPs_vecs.T, 'sqeuclidean').squeeze(0)

            num_dis = np.argmin(ssds)
            min_ssd = ssds[num_dis]
            outlier_rej = True
            Refine = True

            if outlier_rej:
                if ((ssds <= 1.5 * min_ssd).sum() < 3) and (num_dis != 0) and (num_dis != max_disp - min_disp):
                    if Refine:
                        x = np.asarray([num_dis - 1, num_dis, num_dis + 1])
                        p = np.polyfit(x, ssds[x], 2)
                        img_dis[row, col] = max_disp + p[1] / (2 * p[0]) - 1
                    else:
                        img_dis[row, col] = float(-(num_dis - max_disp + 1))
            else:
                img_dis[row, col] = -(num_dis - max_disp + 1)

    return img_dis