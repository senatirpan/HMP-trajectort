#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys  
sys.path.insert(0, '/Users/SenaTirpan/Desktop/gitlab-ipvs/humoro')

from humoro.trajectory import Trajectory

st = Trajectory()

st.loadTrajHDF5("/Users/SenaTirpan/Desktop/gitlab-ipvs/humoro/mogaze/p1_1_human_data.hdf5")

sub_traj = st.subTraj(3000,3360)


# In[2]:


print(sub_traj.data.shape[0])


# In[3]:


from humoro.player_pybullet import Player

pp = Player()

pp.spawnHuman("Human1")
pp.addPlaybackTraj(sub_traj, "Human1")

pp.play(duration=360, startframe=3000)


# In[4]:


from humoro.kin_pybullet import HumanKin

kinematic = HumanKin()


# In[5]:


baseTransY_id = kinematic.inv_index["baseTransY"]
print("baseTransY position Vec3:")
kinematic.get_position(baseTransY_id)


# In[6]:


baseTransY_id = kinematic.inv_index["baseTransY"]
for i in range (sub_traj.data.shape[0]):
    kinematic.set_state(sub_traj,i)
    print(kinematic.get_position(baseTransY_id))    


# In[7]:


sys.path.insert(0, '/Users/SenaTirpan/Desktop/gitlab-ipvs/humoro/examples/human_robot_trajopt')
import helpers

import numpy as np
from helpers import SDF

baseTransY_id = kinematic.inv_index["baseTransY"]
vector = np.array(kinematic.get_position(baseTransY_id))
matrix = np.zeros((sub_traj.data.shape[0],2))
print(vector)

print(matrix.shape)
#print(vector.data[0])
#print(vector.data[1])

#matrix[0,0] = vector.data[0]

for i in range(sub_traj.data.shape[0]):
    kinematic.set_state(sub_traj,i)
    vector = np.array(kinematic.get_position(baseTransY_id))
    for j in range(2):
        matrix[i,j] = vector.data[j]
        #matrix[i,j] = vector.data[j]*helpers.SDF.px_per_m        

min_X = 0;

for i in range(sub_traj.data.shape[0]):
    if(matrix[i][0] < min_X):
        min_X = matrix[i][0]

min_Y = 0;

for i in range(sub_traj.data.shape[0]):
    if(matrix[i][1] < min_Y):
        min_Y = matrix[i][1]

#print(matrix[0][i].min())  # -1.057856798171997    
print("--------")
print(min_X)  # -1.057856798171997

print("--------")
print(min_Y)  # -0.43058595061302185

for i in range(sub_traj.data.shape[0]):
    matrix[i][0] += (-1*min_X)
    matrix[i][1] += (-1*min_Y)
    
print(matrix.min()) # 0.0

min_X2 = 150;

for i in range(sub_traj.data.shape[0]):
    if(matrix[i][0] > min_X2):
        min_X2 = matrix[i][0]

min_Y2 = 150;

for i in range(sub_traj.data.shape[0]):
    if(matrix[i][1] > min_Y2):
        min_Y2 = matrix[i][1]

print("new mins")
print(min_X2) # 0.0
print(min_Y2) #0.0

print(matrix.shape)  
print(matrix)

for i in range(sub_traj.data.shape[0]):
    #for j in range(2):
        #matrix[i,j] *= helpers.SDF.px_per_m 
    matrix[i,:] = helpers.SDF.m_to_pix(matrix[i,:])

print("after the pixel world")
print(matrix)

for i in range(sub_traj.data.shape[0]):
    for j in range(2):
        matrix[i,j] = int(matrix[i,j])
                        
print("after int() for index")
print(matrix)


# In[8]:


import numpy as np
import matplotlib.pyplot as plt

baseTransX = [row[0] for row in matrix]
baseTransY = [row[1] for row in matrix]

x = baseTransX

for i in range(len(baseTransX)):
    baseTransX[i] = 150 - baseTransX[i]
    
print(max(baseTransX))
print(min(baseTransX))

y = baseTransY

plt.xlabel('baseTransX') 
plt.ylabel('baseTransY') 

plt.plot(x, y)
plt.grid()
plt.show()


# In[2]:


from humoro.load_scenes import autoload_objects

obj_trajs, obj_names = autoload_objects(pp, "/Users/SenaTirpan/Desktop/gitlab-ipvs/humoro/mogaze/p1_1_object_data.hdf5", "/Users/SenaTirpan/Desktop/gitlab-ipvs/humoro/mogaze/scene.xml")

pp.play(duration=360, startframe=3000)
img = pp.getFrame2d(3100)

print(type(img)) 
plt.grid()
plt.imshow(img)


# In[50]:


print(img)

#imgWithTraj = img

#print(imgWithTraj)

print(matrix.shape)

for i in range(sub_traj.data.shape[0]):
    j = 0
    #x = int(matrix[i, j])
    #y = int(matrix[i, j+1])
    x = int(baseTransX[i])
    y = int(baseTransY[i])
    if(y == 150 or x == 150):
        continue
    img[x, y] = 2

print(img)
plt.grid()
plt.imshow(img)


# In[ ]:





# In[ ]:




