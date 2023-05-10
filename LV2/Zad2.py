import numpy as np
import matplotlib.pyplot as plt

arr = np.loadtxt(open("data.csv"),delimiter=",", skiprows=1)
print("Mjerenja su izvrsena na "+str(arr.shape[0])+" osoba")   

height = arr[:,1]
weight = arr[:,2]

h2=height[::50]
w2=weight[::50]


plt.scatter(height, weight)
plt.axis([100, 230 , 0, 230])
plt.show()  


plt.scatter(h2, w2)
plt.axis([100, 230 , 0, 230])
plt.show()  


print("Max visina: "+ str(max(height)))
print("Min visina: "+ str(min(height)))
print("Srednja visina: "+ str(np.mean(height)))


male = (arr[:,0] == 1)
female = (arr[:,0] == 0)

print("Minimalna visina muškaraca:",arr[male,1].min(),"cm")
print("Maksimalna visina muškaraca:",arr[male,1].max(),"cm")
print("Prosječna visina muškaraca:",arr[male,1].mean(),"cm")