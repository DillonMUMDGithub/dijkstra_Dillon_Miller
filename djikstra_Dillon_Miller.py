""" Main Fuction for running project 2 
    """
    
#Import Necessary dependencies
import matplotlib.pyplot as plt
import time as time
from queue import PriorityQueue
import numpy as np
import random
import os
import cv2 as cv


    
#Function to move Up, adds C2C    
def MoveUp(Node,C2C):
    NewNode = Node.copy()
    NewNode[1] = NewNode[1]+1
    NewCost = C2C+1
    return(NewNode,NewCost)
#Function to move Down, adds C2C
def MoveDown(Node,C2C):
    NewNode = Node.copy()
    NewNode[1] = NewNode[1]-1
    NewCost = C2C+1
    return(NewNode,NewCost)
#Function to move Left, adds C2C
def MoveLeft(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]-1
    NewCost = C2C+1
    return(NewNode,NewCost)
#Function to move Right, adds C2C
def MoveRight(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]+1
    NewCost = C2C+1
    return(NewNode,NewCost)
#Function to move Up and Left, adds C2C
def MoveUL(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]-1
    NewNode[1] = NewNode[1]+1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
#Function to move Up and Right, adds C2C
def MoveUR(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]+1
    NewNode[1] = NewNode[1]+1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
#Function to move Down and Left, adds C2C
def MoveDL(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]-1
    NewNode[1] = NewNode[1]-1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
#Function to move Down and Right, adds C2C
def MoveDR(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]+1
    NewNode[1] = NewNode[1]-1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
#Function to check if the current node is the goal node
def CheckifGoal(Node,GoalNode):
    Stop = 0
    if Node ==GoalNode:
        Stop = 1
    return Stop
#function to check if the node is an obstacle or outside the boundaries
def CheckIfObstacle(Node, ObstMat):
    InObstacle = 0
    if Node[1]>499:
        InObstacle = 1
    elif Node[1]<0:
        InObstacle = 1    
    elif Node[0]<0:
        InObstacle = 1 
    elif Node[0]>1199:
        InObstacle = 1
    elif ObstMat[Node[1]][Node[0]] == -1:
        InObstacle = 1   
    return InObstacle
#Function to check if the node is within the closed node
def CheckClosed(Node,Closed):
    InClosed = 0
    if Node in Closed:
        InClosed = 1
    return InClosed
##### Definition of the Map #####
#Map matrices, BW and Color
ObstMat = np.full((500,1200),np.inf)
ObstMatR = np.full((500,1200),0)
ObstMatG = np.full((500,1200),0)
ObstMatB = np.full((500,1200),0)

#Leftmost object,-1 value for C2C storage, Red in color for bloating
print(ObstMat.shape)
for i in range(94,180):
    for k in range(89,500):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 255
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 0
#Leftmost boundary, blue for primary obstacle
for i in range(99,175):
    for k in range(94,500):
        ObstMat[k][i] = -1
        ObstMatB[k][i] = 255
        ObstMatR[k][i] = 0
        
#Right rectangle object,-1 value for C2C storage, Red in color for bloating
for i in range(269,354):
    for k in range(0,404):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 255
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 0
# Right rectangle obstacle, blue for primary obstacle
for i in range(274,349):
    for k in range(0,399):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 0
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 255        
        
#right U shape obstacle
# -1 value for C2C storage, Red in color for bloating
#Primary vertical rectangle
for i in range(1014,1104):
    for k in range(44,454):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 255
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 0
for i in range(1019,1099):
    for k in range(49,449):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 0
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 255
#Top Horizonal rectangle
for i in range(894,1019):
    for k in range(369,454):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 255
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 0
for i in range(899,1019):
    for k in range(374,449):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 0
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 255 
#Bottom Horizontal Rectangle     
for i in range(894,1019):
    for k in range(44,124):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 255
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 0
for i in range(899,1019):
    for k in range(49,119):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 0
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 255        
        
#Hexagon obstacle
#Establishes one central rectangle, and then 4 triangles
#MiddleRectangle:
for i in range(514,784):
    for k in range(174,324):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 255
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 0
#Red Barrier Middle Rectangle
for i in range(519,779):
    for k in range(174,324):
        ObstMat[k][i] = -1
        ObstMatR[k][i] = 0
        ObstMatG[k][i] = 0
        ObstMatB[k][i] = 255       
        
#TopLeftTriangle
GRat = 80/135
for i in range(514,649):
    for k in range(324,404):
        # ObstMat[k][i] = -1
        XTemp = i-514
        YTemp = k-324
        if (XTemp*GRat)>YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 255
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 0 
#Top Left No Barrier
GRat = 75/130
for i in range(519,649):
    for k in range(324,399):
        # ObstMat[k][i] = -1
        XTemp = i-519
        YTemp = k-324
        if (XTemp*GRat)>YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 0
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 255            
               
#TopRight Triangle
GRat = -80/135          
for i in range(649,784):
    for k in range(324,404):
        # ObstMat[k][i] = -1
        XTemp = i-649
        YTemp = k-324
        if (XTemp*GRat)+80>YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 255
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 0 
#TopRight No barrier
GRat = -75/130        
for i in range(649,779):
    for k in range(324,399):
        # ObstMat[k][i] = -1
        XTemp = i-649
        YTemp = k-324
        if (XTemp*GRat)+75>YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 0
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 255        
#BotRightTriangle
GRat = 80/135
for i in range(649,784):
    for k in range(94,174):
        # ObstMat[k][i] = -1
        XTemp = i-649
        YTemp = k-94
        if (XTemp*GRat)<YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 255
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 0 
# BotLeft No fluffing
GRat = 75/130
for i in range(649,779):
    for k in range(99,174):
        # ObstMat[k][i] = -1
        XTemp = i-649
        YTemp = k-99
        if (XTemp*GRat)<YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 0
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 255 
#BotLeft Triangle
GRat = -80/135          
for i in range(514,649):
    for k in range(94,174):
        XTemp = i-514
        YTemp = k-94
        if (XTemp*GRat)+80<YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 255
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 0    
#Bot Left No fluffing 
GRat = -75/130        
for i in range(519,649):
    for k in range(99,174):
        XTemp = i-519
        YTemp = k-99
        if (XTemp*GRat)+75<YTemp:
            ObstMat[k][i] = -1
            ObstMatR[k][i] = 0
            ObstMatG[k][i] = 0
            ObstMatB[k][i] = 255
while(True):
    A = input("Enter r for random start and end points or u for user input start and end points")
    if A == 'r':
        startNode = [random.randint(0,50),random.randint(50,450)]
        endNode = [random.randint(1050,1149),random.randint(50,450)]
        break
    elif A == 'u':
        Xstart = int(input("enter an Integer value to for the x value of the start "))
        Ystart = int(input("enter an Integer value to for the y value of the start "))
        Xend = int(input("enter an Integer value to for the x value of the end "))
        Yend = int(input("enter an Integer value to for the y value of the end "))
        startNode = [Xstart,Ystart]
        endNode = [Xend,Yend]
        if ObstMat[Ystart][Xstart] == -1:
            print("start node within an obstacle ")
        elif ObstMat[Yend][Xend] == -1:
            print("end node within an obstacle ")
        else:
            break
        
    else:
        print("invalid input")
#Starting C2C values
OriginalC2C = 0
WorkingC2C = 0

OpenQ = []
ClosedQ = []
ClosedC2C = []
VisitedQ = []
ClosedParent = []
Parent = []
C2C = []
OpenQ.append(startNode)
C2C.append(0)
TotalQ = PriorityQueue()
TotalQ.put((0, [startNode,'N/A']))
ClosedQPrio = PriorityQueue()
#Create a matrix to store images for the video creation
imgmat = []
#first parent is N/A
Parent.append('N/A')
#Conditionalcounter to track progess
Go = 1
#Conditional to keep the while loop going
Stop = 0
#Start the timer
start = time.time()
while(Stop==0):
    try:
        Go=Go+1

        
        QPop = TotalQ.get()
        WorkingC2C = QPop[0]
        WorkingNode = QPop[1][0]
        WorkingParent = QPop[1][1]
        Stop = CheckifGoal(WorkingNode,endNode)
        if Stop ==1:
            FinalNode = WorkingNode
            FinalParent = WorkingParent
            FinalC2C = WorkingC2C
            break

        Up,UpC2C = MoveUp(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(Up,ObstMat)
        if A ==0:
            B = CheckClosed(Up,ClosedQ)
            if B ==0:
                if UpC2C<ObstMat[Up[1]][Up[0]]:
                    ObstMat[Up[1]][Up[0]] = UpC2C
                    ObstMatG[Up[1]][Up[0]] = 255
                    OpenQ.append(Up)
                    Parent.append(WorkingNode)
                    C2C.append(UpC2C)
                    TotalQ.put((UpC2C,[Up,WorkingNode]))   

            
        Down,DownC2C =MoveDown(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(Down,ObstMat)
        if A ==0:
            B = CheckClosed(Down,ClosedQ)
            if B ==0:
                if DownC2C<ObstMat[Down[1]][Down[0]]:
                    ObstMat[Down[1]][Down[0]] = DownC2C
                    ObstMatG[Down[1]][Down[0]] = 255
                    OpenQ.append(Down)
                    Parent.append(WorkingNode)
                    C2C.append(DownC2C)
                    TotalQ.put((DownC2C,[Down,WorkingNode])) 
                    
        Left,LeftC2C =MoveLeft(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(Left,ObstMat)
        if A ==0:
            B = CheckClosed(Left,ClosedQ)
            if B ==0:
                if LeftC2C<ObstMat[Left[1]][Left[0]]:
                    ObstMat[Left[1]][Left[0]] = LeftC2C
                    ObstMatG[Left[1]][Left[0]] = 255
                    OpenQ.append(Left)
                    Parent.append(WorkingNode)
                    C2C.append(LeftC2C)
                    TotalQ.put((LeftC2C,[Left,WorkingNode])) 
                    
        Right,RightC2C =MoveRight(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(Right,ObstMat)
        if A ==0:
            B = CheckClosed(Right,ClosedQ)
            if B ==0:
                if RightC2C<ObstMat[Right[1]][Right[0]]:
                    ObstMat[Right[1]][Right[0]] = RightC2C
                    ObstMatG[Right[1]][Right[0]] = 255
                    OpenQ.append(Right)
                    Parent.append(WorkingNode)
                    C2C.append(RightC2C)
                    TotalQ.put((RightC2C,[Right,WorkingNode])) 
                    
        UL,ULC2C =MoveUL(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(UL,ObstMat)
        if A ==0:
            B = CheckClosed(UL,ClosedQ)
            if B ==0:
                if ULC2C<ObstMat[UL[1]][UL[0]]:
                    ObstMat[UL[1]][UL[0]] = ULC2C
                    ObstMatG[UL[1]][UL[0]] = 255
                    OpenQ.append(UL)
                    Parent.append(WorkingNode)
                    C2C.append(ULC2C)
                    TotalQ.put((ULC2C,[UL,WorkingNode])) 
                    
        UR,URC2C =MoveUR(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(UR,ObstMat)
        if A ==0:
            B = CheckClosed(UR,ClosedQ)
            if B ==0:
                if URC2C<ObstMat[UR[1]][UR[0]]:
                    ObstMat[UR[1]][UR[0]] = URC2C
                    ObstMatG[UR[1]][UR[0]] = 255
                    OpenQ.append(UR)
                    Parent.append(WorkingNode)
                    C2C.append(URC2C)
                    TotalQ.put((URC2C,[UR,WorkingNode])) 
                    
        DL,DLC2C =MoveDL(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(DL,ObstMat)
        if A ==0:
            B = CheckClosed(DL,ClosedQ)
            if B ==0:
                if DLC2C<ObstMat[DL[1]][DL[0]]:
                    ObstMat[DL[1]][DL[0]] = DLC2C
                    ObstMatG[DL[1]][DL[0]] = 255
                    OpenQ.append(DL)
                    Parent.append(WorkingNode)
                    C2C.append(DLC2C)
                    TotalQ.put((DLC2C,[DL,WorkingNode])) 
                    
        DR,DRC2C =MoveDR(WorkingNode,WorkingC2C)
        A = CheckIfObstacle(DR,ObstMat)
        if A ==0:
            B = CheckClosed(DR,ClosedQ)
            if B ==0:
                if DRC2C<ObstMat[DR[1]][DR[0]]:
                    ObstMat[DR[1]][DR[0]] = DRC2C
                    ObstMatG[DR[1]][DR[0]] = 255
                    OpenQ.append(DR)
                    Parent.append(WorkingNode)
                    C2C.append(DRC2C)
                    TotalQ.put((DRC2C,[DR,WorkingNode])) 
        if Go%1500 ==0:
            # print(Go)
            ObstMat3d = np.dstack((ObstMatR,ObstMatG,ObstMatB))
            imgname = "Temp"+str(Go)+".png"
            cv.imwrite(imgname, ObstMat3d)
            img = cv.imread(imgname)
            imgmat.append(img)
            os.remove(imgname) 
            if Go%15000 ==0:
                print("Nodes Searched:", Go)
            # video.write(img)
            # os.remove("Tempimg.png") 
            
        
    except:
        break
# video.release()
# os.remove("Tempimg.png") 
end = time.time()    
print("Time Taken To Complete:",end-start)    


# Least = TotalQ.get()
# print(Least)
LeastC2C = FinalC2C
LeastNode = FinalNode
LeastParent = FinalParent
print("Final Cost to Come:",WorkingC2C)
# print(WorkingNode)
# print(WorkingParent)

# print(len(Parent))
# print(len(OpenQ))
# B = TotalQ.get([LeastParent])
# print(B)
TrackBack = []
# B = OpenQ.index(LeastParent)
# print(B)
# C = OpenQ[B]
end=1
print("computing optimal path to final node")
while (end==1):
    B = OpenQ.index(LeastParent)  
    C = OpenQ[B]
    TrackBack.append(C)
    D = Parent[B]
    LeastParent = D
    if D == [0,0]:
        end = 0
    elif D == startNode:    
        end = 0
TrackBack.append(startNode)

counttrack = 0
for pixel in TrackBack:
    try:
        ObstMat[pixel[1]][pixel[0]] = -1
        ObstMat[pixel[1]-1][pixel[0]] = -1
        ObstMat[pixel[1]+1][pixel[0]] = -1
        ObstMat[pixel[1]][pixel[0]-1] = -1
        ObstMat[pixel[1]][pixel[0]+1] = -1
        ObstMatR[pixel[1]][pixel[0]] = -1
        ObstMatR[pixel[1]-1][pixel[0]] = 255
        ObstMatR[pixel[1]+1][pixel[0]] = 255
        ObstMatR[pixel[1]][pixel[0]-1] = 255
        ObstMatR[pixel[1]][pixel[0]+1] = 255
        ObstMatG[pixel[1]][pixel[0]] = 255
        ObstMatG[pixel[1]-1][pixel[0]] = 255
        ObstMatG[pixel[1]+1][pixel[0]] = 255
        ObstMatG[pixel[1]][pixel[0]-1] = 255
        ObstMatG[pixel[1]][pixel[0]+1] = 255
        ObstMatB[pixel[1]][pixel[0]] = 255
        ObstMatB[pixel[1]-1][pixel[0]] = 255
        ObstMatB[pixel[1]+1][pixel[0]] = 255
        ObstMatB[pixel[1]][pixel[0]-1] = 255
        ObstMatB[pixel[1]][pixel[0]+1] = 255
        counttrack = counttrack+1
        #Append the trackback iamges to the 
        if counttrack%20 ==0:
            ObstMat3d = np.dstack((ObstMatR,ObstMatG,ObstMatB))
            imgname = "Temp"+str(Go)+".png"
            cv.imwrite(imgname, ObstMat3d)
            img = cv.imread(imgname)
            imgmat.append(img)
            os.remove(imgname) 
    except:
        break
    
 #Append 10 frames of the final path (1second)   
for i in range (0,10):
    ObstMat3d = np.dstack((ObstMatR,ObstMatG,ObstMatB))
    imgname = "Temp"+str(Go)+".png"
    cv.imwrite(imgname, ObstMat3d)
    img = cv.imread(imgname)
    imgmat.append(img)
    os.remove(imgname)    
    
print("Creating Final Output Video")      
video=cv.VideoWriter('TestVideo.mp4',cv.VideoWriter_fourcc(*'MP4V'),10,(1200,500))
for i in range(0,len(imgmat)):
    video.write(imgmat[i])
video.release()
cv.destroyAllWindows()

ObstMat3d = np.dstack((ObstMatR,ObstMatG,ObstMatB))
cv.imwrite("FinalMap.png", ObstMat3d)
plt.matshow(ObstMat3d)
plt.show()


# plt.matshow(ObstMat,cmap = 'gray')
# plt.show()