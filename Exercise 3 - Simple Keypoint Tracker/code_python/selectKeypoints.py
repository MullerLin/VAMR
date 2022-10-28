import numpy as np


def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
    the current maximum.
    """
    max_value = []
    index_stack_x = []
    index_stack_y = []
    for i in range(num):
        max_index = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        max_value.append(scores[max_index])
        m, n =  max_index
        if max_value[i] > 0:
            index_stack_x.append(m)
            index_stack_y.append(n)
        else:
            break
        scores[m-r:m+r+1, n-r:n+r+1] = 0

    return np.array([index_stack_x, index_stack_y])