""" Main Fuction for running project 2 
    """
    
#Establish Obstacle Matrix    
from ObstacleMatTest import ObstMat
import matplotlib.pyplot as plt
import time as time
# import heapq
from queue import PriorityQueue
    
def MoveUp(Node,C2C):
    NewNode = Node.copy()
    NewNode[1] = NewNode[1]+1
    NewCost = C2C+1
    return(NewNode,NewCost)
def MoveDown(Node,C2C):
    NewNode = Node.copy()
    NewNode[1] = NewNode[1]-1
    NewCost = C2C+1
    return(NewNode,NewCost)
def MoveLeft(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]-1
    NewCost = C2C+1
    return(NewNode,NewCost)
def MoveRight(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]+1
    NewCost = C2C+1
    return(NewNode,NewCost)
def MoveUL(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]-1
    NewNode[1] = NewNode[1]+1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
def MoveUR(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]+1
    NewNode[1] = NewNode[1]+1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
def MoveDL(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]-1
    NewNode[1] = NewNode[1]-1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
def MoveDR(Node,C2C):
    NewNode = Node.copy()
    NewNode[0] = NewNode[0]+1
    NewNode[1] = NewNode[1]-1
    NewCost = C2C+1.4
    return(NewNode,NewCost)
def CheckifGoal():


    return
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

def CheckAllClosed(Nodes,Closed):
    InClosedMat = [0,0,0,0,0,0,0,0]
    for ClosedMat in Closed:
            if Nodes[0]==ClosedMat:
                InClosedMat[0] = 1
            if Nodes[1]==ClosedMat:
                InClosedMat[1] = 1
            if Nodes[2]==ClosedMat:
                InClosedMat[2] = 1
            if Nodes[3]==ClosedMat:
                InClosedMat[3] = 1
            if Nodes[4]==ClosedMat:
                InClosedMat[4] = 1               
            if Nodes[5]==ClosedMat:
                InClosedMat[5] = 1
            if Nodes[6]==ClosedMat:
                InClosedMat[6] = 1
            if Nodes[7]==ClosedMat:
                InClosedMat[7] = 1
          
    
    return InClosedMat

startNode = [0,0]
endNode = [1200,500]
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

Parent.append('N/A')
Go = 1


# print(CheckIfObstacle([150,50],ObstMat))
# plt.ion()
# fig = plt.figure()
# fig = plt.figure()
# plt.matshow(ObstMat)
# plt.show(block=False)
#10001 - 5.688
#100001 - 1212
# For about half: 1000001
# Number of pixels 600,000
start = time.time()
# QPop = TotalQ.get()
# WorkingC2C = QPop[0]
# WorkingNode = QPop[1][0]
# WorkingParent = QPop[1][1]
# print(WorkingC2C)





while(Go<10001):
    # plt.show(block=False)
    
    Go=Go+1
    print(Go)
    # print('while')
    # C2Cindex = min(C2C)
    # WorkingNode = OpenQ.pop(C2Cindex)
    # WorkingC2C = C2C.pop(C2Cindex)
    # WorkingParent = Parent.pop(C2Cindex)
    QPop = TotalQ.get()
    WorkingC2C = QPop[0]
    WorkingNode = QPop[1][0]
    WorkingParent = QPop[1][1]
    # print(WorkingNode,WorkingC2C)
    ClosedQ.append(WorkingNode)
    ClosedParent.append(WorkingParent)
    ClosedC2C.append(WorkingC2C)
    TotalQ.put((WorkingC2C,[WorkingNode,WorkingParent]))   
       
    Up,UpC2C = MoveUp(WorkingNode,WorkingC2C)
    Down,DownC2C =MoveDown(WorkingNode,WorkingC2C)
    Left,LeftC2C =MoveLeft(WorkingNode,WorkingC2C)
    Right,RightC2C =MoveRight(WorkingNode,WorkingC2C)
    UL,ULC2C =MoveUL(WorkingNode,WorkingC2C)
    UR,URC2C =MoveUR(WorkingNode,WorkingC2C)
    DL,DLC2C =MoveDL(WorkingNode,WorkingC2C)
    DR,DRC2C =MoveDR(WorkingNode,WorkingC2C)
    A0 = CheckIfObstacle(Up,ObstMat)
    A1 = CheckIfObstacle(Down,ObstMat)
    A2 = CheckIfObstacle(Left,ObstMat)
    A3 = CheckIfObstacle(Right,ObstMat)
    A4 = CheckIfObstacle(UL,ObstMat)
    A5 = CheckIfObstacle(UR,ObstMat)
    A6 = CheckIfObstacle(DL,ObstMat)
    A7 = CheckIfObstacle(DR,ObstMat)
    # InObstacleMat = [A0,A1,A2,A3,A4,A5,A6,A7]
    AllNewNodes = [Up,Down,Left,Right,UR,UL,DL,DR]

    B = CheckAllClosed(AllNewNodes,ClosedQ)   
       
    
    if A0 ==0:
        # B = CheckClosed(Up,ClosedQ)
        if B[0] ==0:
            if UpC2C<ObstMat[Up[1]][Up[0]]:
                ObstMat[Up[1]][Up[0]] = UpC2C
                OpenQ.append(Up)
                Parent.append(WorkingNode)
                C2C.append(UpC2C)
          
    
    # A = CheckIfObstacle(Down,ObstMat)
    if A1 ==0:
        # B[1] = CheckClosed(Down,ClosedQ)
        if B[1] ==0:
            if DownC2C<ObstMat[Down[1]][Down[0]]:
                ObstMat[Down[1]][Down[0]] = DownC2C
                OpenQ.append(Down)
                Parent.append(WorkingNode)
                C2C.append(DownC2C)
                
    # Left,LeftC2C =MoveLeft(WorkingNode,WorkingC2C)
    # A = CheckIfObstacle(Left,ObstMat)
    if A2 ==0:
        # B = CheckClosed(Left,ClosedQ)
        if B[2] ==0:
            if LeftC2C<ObstMat[Left[1]][Left[0]]:
                ObstMat[Left[1]][Left[0]] = LeftC2C
                OpenQ.append(Left)
                Parent.append(WorkingNode)
                C2C.append(LeftC2C)
                
    # Right,RightC2C =MoveRight(WorkingNode,WorkingC2C)
    # A = CheckIfObstacle(Right,ObstMat)
    if A3 ==0:
        # B = CheckClosed(Right,ClosedQ)
        if B[3] ==0:
            if RightC2C<ObstMat[Right[1]][Right[0]]:
                ObstMat[Right[1]][Right[0]] = RightC2C
                OpenQ.append(Right)
                Parent.append(WorkingNode)
                C2C.append(RightC2C)
                
    # UL,ULC2C =MoveUL(WorkingNode,WorkingC2C)
    # A = CheckIfObstacle(UL,ObstMat)
    if A4 ==0:
        # B = CheckClosed(UL,ClosedQ)
        if B[4] ==0:
            if ULC2C<ObstMat[UL[1]][UL[0]]:
                ObstMat[UL[1]][UL[0]] = ULC2C
                OpenQ.append(UL)
                Parent.append(WorkingNode)
                C2C.append(ULC2C)
                
    # UR,URC2C =MoveUR(WorkingNode,WorkingC2C)
    # A = CheckIfObstacle(UR,ObstMat)
    if A5 ==0:
        # B = CheckClosed(UR,ClosedQ)
        if B[5] ==0:
            if URC2C<ObstMat[UR[1]][UR[0]]:
                ObstMat[UR[1]][UR[0]] = URC2C
                OpenQ.append(UR)
                Parent.append(WorkingNode)
                C2C.append(URC2C)
                
    # DL,DLC2C =MoveDL(WorkingNode,WorkingC2C)
    # A = CheckIfObstacle(DL,ObstMat)
    if A6 ==0:
        # B = CheckClosed(DL,ClosedQ)
        if B[6] ==0:
            if DLC2C<ObstMat[DL[1]][DL[0]]:
                ObstMat[DL[1]][Up[0]] = DLC2C
                OpenQ.append(DL)
                Parent.append(WorkingNode)
                C2C.append(DLC2C)
                
    # DR,DRC2C =MoveDR(WorkingNode,WorkingC2C)
    # A = CheckIfObstacle(DR,ObstMat)
    if A7 ==0:
        # B = CheckClosed(DR,ClosedQ)
        if B[7] ==0:
            if DRC2C<ObstMat[DR[1]][DR[0]]:
                ObstMat[DR[1]][DR[0]] = DRC2C
                OpenQ.append(DR)
                Parent.append(WorkingNode)
                C2C.append(DRC2C)
end = time.time()    
print(end-start)           
plt.matshow(ObstMat)
plt.show()

#10001 - 33
#20001 - 129


#New
#10001 - 20.49
#20001 - 80