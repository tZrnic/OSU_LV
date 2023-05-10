import numpy as np
import matplotlib.pyplot as plt

bijeli = np.ones((50, 50), dtype = np.uint8) * 255
crni = np.zeros((50, 50), dtype = np.uint8)
gornji = np.hstack((crni, bijeli))
donji = np.hstack((bijeli, crni))
plt.imshow(np.vstack((gornji, donji)), cmap='gray')
plt.show()