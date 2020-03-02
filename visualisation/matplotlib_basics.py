import matplotlib.pyplot as plt
import numpy as np

'''
It is possible to change the appearance of a plot overall by setting the style
with pyplot, rather than changing individual elements of a plot as you go.
You can do with the following syntax.
'''
plt.style.use('ggplot')

'''
Changing the style in this way changes the appearance of all visualisations 
in the session, and will remain this way until you change it again. 

There are many things that you can consider when you are choosing a style. 
If colour is important then consider a colourblind palette, if it will be printed
then use lighter backgrounds to avoid ink use and grayscale when you know the 
visualisation is going to be used in black and white (i.e. printed not in colour)
'''

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

'''
You can add labels and titles to the plot using the set functions on ax
'''

ax.set_xlabel('Numbers (X)')
ax.set_ylabel('Numbers (Y)')
ax.set_title('Example plot with line styling')

plt.show()
