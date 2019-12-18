import numpy as np

x,y=[0,3]
pos=[0,3]
size_x,size_y=[5,5]

alle_nachbarn=np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
pos=pos+alle_nachbarn
print('x')