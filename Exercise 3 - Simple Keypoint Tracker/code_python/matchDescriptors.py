import numpy as np
from scipy.spatial.distance import cdist


def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor. The descriptor vectors are MxQ and MxD where M is the descriptor dimension and Q and D the
    amount of query and database descriptors respectively. matches(i) will be zero(-1) if there is no database descriptor
    with an SSD < lambda * min(SSD). No two non-zero elements of matches will be equal.
    """
    dist = cdist(np.array(query_descriptors.T), np.array(database_descriptors.T), metric='euclidean')
    index = np.argmin(dist, axis=1)
    value = np.min(dist, axis=1)
    d_min = np.min(value)
    for i in range(index.size):
        if value[i] > match_lambda*d_min:
            index[i] = 0

    return index