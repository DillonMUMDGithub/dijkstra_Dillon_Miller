# dijkstra_Dillon_Miller
Github repository for ENPM661 Project 2

Dillon Miller
Dmille19@umd.edu


Code Name: proj2_Dillon_Miller.py

Imported Libraries:

matplotlib
time
queue
numpy
random
os
cv2

Inputs:
This code allows for the user to enter in their own start points or start at a random value within the left side of the map and a random endpoint within the left side of the map. It is assumed that you understand the extent of the graph and its obstacles, however if not, select random. if you are entering your own start and end points, they must be integer values, and if they are within an obstacle, it will not be allowed.

The dimensions of the map are from 0 to 1200 on the x axis and 0 to 500 on the y axis.


Instructions:
In order to run this code, change your working directory in your terminal to the location of the file, and execute >python3 djikstra_Dillon_Miller.py

After execution and reponse to the inital prompts, the program will update you with the progress of the search, ultimately printing the time taken to solve. This will take around 30 to 40 seconds. Afterwards, the trackback will initialize and take some time to find the resulting optimal path.



Outputs:
This program, upon completing will generate a video file, labeled djiksta_Dillon_Miller.mp4, as well as an image file labeled FinalMap.png. The ideo will display the progress of the search algorithm as well as the final path.
