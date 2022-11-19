import csv
import numpy as np


#Saving the X and Y Values in an Array
x_values = []
y_values = []


#df = pd.read_csv("data.csv")

#Opening the File
with open('data.csv','r') as file:
    filecontent=csv.reader(file)

    for line in filecontent:
        x_values.append(line[0])
        y_values.append(line[1])

def closest1(discrete_x, i):
    discrete_x = np.asarray(discrete_x)
    idx = (np.abs(discrete_x - i)).argmin()
    return discrete_x[idx]

def closest2(discrete_y, j):
    discrete_y = np.asarray(discrete_y)
    idy = (np.abs(discrete_y - j)).argmin()
    return discrete_y[idy]

discrete_x = [-300,-200,-150,-100,-50,-25,-10,-5,-1,0,1,5,10,50,100,150,200,300]
discrete_y = [-100,-50,-25,-10,-5,-1,0,1,5,10,25,50,100]

#Append the values to normalize each one

for i in x_values:
    print(closest1(discrete_x, int(i)))

for j in y_values:
    print(closest2(discrete_y, int(j)))

            
