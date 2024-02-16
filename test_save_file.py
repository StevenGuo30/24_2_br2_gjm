import numpy as np

rot_degrees = np.array([1,2,3,4,5,6])

with open('F:\\Soft_arm\\code\\rot_degree.txt','w') as file:
    file.writelines([str(d)+' ' for d in rot_degrees])
    file.write('\n')