import numpy as np
import matplotlib.pyplot as plt

import random
lists = np.random.randint(0,2,(100,100))

plt.xlabel('x')
plt.ylabel('y')

plt.xlim(right=100,left=0)
plt.ylim(top=100, bottom=0)

#lists = [0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0]

x1 = []
y1 = []

for i in range(len(lists)):
    if lists[i] == 1:
        x1.append((i - i%100)/100)
        y1.append(i%100)

#x1 = [0,1,2,3,4]
#y1 = [4,3,2,1,0]
print(x1)
print(y1)

colors1 = '#00CED1'
plt.scatter(x1, y1, c=colors1, alpha=0.4)

plt.show()
