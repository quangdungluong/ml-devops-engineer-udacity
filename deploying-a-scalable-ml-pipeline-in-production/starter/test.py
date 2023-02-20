import pickle
import os
import numpy as np
dir = os.path.dirname(__file__)
print(dir)
lb = pickle.load(open(os.path.join(dir, "model/lb.pkl"), 'rb'))
print(lb.inverse_transform(np.array([1])))