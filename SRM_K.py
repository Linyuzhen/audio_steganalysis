import numpy as np

SRM_k = np.array([[1,-1,0,0,0],
                  [1,-2,1,0,0],
                  [1,-3,3,-1,0],
                  [1,-4,6,-4,1]
])
# SRM_k = SRM_k.reshape((4,1,5))
# print(SRM_k)
np.save('SRM_K.npy',SRM_k)