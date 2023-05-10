import numpy as np
import matplotlib.pyplot as plt

pic = plt.imread("road.jpg")
print(pic.shape)
print(pic.dtype)
pic = pic[:,:,0].copy()

print(pic.shape)
print(pic.dtype)

sel = pic[0:200,0:300]
plt.imshow(sel)
pic2=pic
pic = np.rot90(pic, -1)
plt.figure()

plt.imshow(np.fliplr(pic), cmap="gray")
plt.show()
plt.figure()

plt.imshow(pic2, cmap="gray")
plt.show()
plt.figure()

plt.imshow(pic2/2, cmap="gray")
plt.show()