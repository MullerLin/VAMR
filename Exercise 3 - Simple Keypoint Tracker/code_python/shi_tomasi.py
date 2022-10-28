import numpy as np
from scipy import signal


def shi_tomasi(img, patch_size):

    sobel_x = np.array((-1, 0, 1, -2, 0, 2, -1, 0, 1)).reshape(3, 3)
    sobel_y = np.array((-1, -2, -1, 0, 0, 0, 1, 2, 1)).reshape(3, 3)
    I_x = signal.convolve2d(img, sobel_x, mode='valid')
    I_y = signal.convolve2d(img, sobel_y, mode='valid')

    I_x_square = I_x * I_x
    I_y_square = I_y * I_y
    I_xy = I_x * I_y

    box_matrix = np.ones((patch_size, patch_size))

    sum_I_xx = signal.convolve2d(I_x_square, box_matrix, mode='valid')
    sum_I_yy = signal.convolve2d(I_y_square, box_matrix, mode='valid')
    sum_I_xy = signal.convolve2d(I_xy, box_matrix, mode='valid')

    m, n = sum_I_xx.shape[0], sum_I_xx.shape[1]

    M1 = np.concatenate((sum_I_xx.reshape(m, n, 1, 1), sum_I_xy.reshape(m, n, 1, 1)), axis=3)
    M2 = np.concatenate((sum_I_xy.reshape(m, n, 1, 1), sum_I_yy.reshape(m, n, 1, 1)), axis=3)
    M = np.concatenate((M1, M2), axis=2)
    w, v = np.linalg.eig(M)
    R_st = np.min(w, axis=2)

    return R_st