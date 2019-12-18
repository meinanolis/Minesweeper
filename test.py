import numpy as np

x,y=[1,3]
pos=[1,3]
size_x,size_y=[5,5]

feld=np.zeros((size_x,size_y))

alle_nachbarn=np.array([[0,1],[0,-1],[1,0],[-1,0],[1,-1],[-1,-1],[-1,1],[1,1]])
pos=pos+alle_nachbarn
pos=pos.T
feld[pos[0],pos[1]]=1
print('x')