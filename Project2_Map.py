import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.collections import PatchCollection
import numpy as np

#Main Map Space
YlimMap = 500.00
XLimMap =1200.00

#Left Most Rectangle (Rectangle 1)
Rl1Ht = 400.00
Rl1Wd = 75.00

#Right Most Rectangle (Rectangle 2)
Rl2Ht = 400.00
Rl2Wd = 75.00

#Last object
Corner1 = 900,50
Corner2 = 900,375
Corner3 = 1020,50

SmRWd = 120
SmRHt = 75
LRWd = 80
LRHt = 400


fig, ax = plt.subplots()
ax.set(xlim=(0,XLimMap),ylim=(0,YlimMap))

Rectangle1 = ptc.Rectangle([100,100],Rl1Wd,Rl1Ht,facecolor='r')
Rectangle2 = ptc.Rectangle([275,0],Rl2Wd,Rl2Ht,facecolor='r')
Rec1 = ptc.Rectangle(Corner1,SmRWd,SmRHt,facecolor='r')
Rec2 = ptc.Rectangle(Corner2,SmRWd,SmRHt,facecolor='r')
Rec3 = ptc.Rectangle(Corner3,LRWd,LRHt,facecolor='r')
Hexagon = ptc.RegularPolygon((650,250), numVertices=6, radius=np.sqrt(3)*75, orientation=0, facecolor='r')




# pc = PatchCollection(Rectangle, facecolor='r')
ax.add_patch(Rectangle1)
ax.add_patch(Rectangle2)
ax.add_patch(Hexagon)
ax.add_patch(Rec1)
ax.add_patch(Rec2)
ax.add_patch(Rec3)

plt.show()

