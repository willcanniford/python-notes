import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

'''
Line graphs with matplotlib have 3 basic features that you can change:
1. Colour
2. Marker
3. Linestyle 

They are controlled and set respectively with `color`, `marker` and `linestyle` arguments. 
'''

numbers = [x for x in range(1, 10)]
fig, ax = plt.subplots()

ax.plot(numbers, color='red', marker='v', linestyle='--')
plt.show()
