import numpy as np
from data_handler import *
from embedding_helper import *

def projection(vectors, direction_vector):
    return np.dot(vectors, direction_vector)

def debias(vectors, direction_vector, lambd):
    proj = projection(vectors, direction_vector)
    return vectors - (lambd * direction_vector * proj.T).T