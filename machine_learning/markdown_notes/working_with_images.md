# Working with images 
Below is some example code that shows the basics of working with images. Often you need the image to be a 1D array to work with for machine learning purposes, but you can then reshape this back into the original shape and use `matplotlib` to display the grayscale image. 

```python 
# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0, :]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13, 8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
```

This could even be turned into a function for working with many samples: 

```python 
def show_as_image(sample, shape):
    # shape = tuple e.g. (13,8)
    bitmap = sample.reshape(shape)
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
```
