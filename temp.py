import numpy as np 


a=np.array([[1,0,0],[0,1,0],[0,0,1]])

print(np.argwhere(a==np.min(a)))

change