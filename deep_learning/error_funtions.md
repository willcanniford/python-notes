# Error functions

Error functions are a defined method that we we decide how close we are to the ideal outcome. We generally use gradient descent to minimise this, with the derivative providing a good way of fine-tuning the parameters with the goal of finding a global mimimum for the error metric.  

There are many error functions out there, and there is one obvious condition that we must have for using gradient descent and that is the that the error function must be __continuous__ in order for us to derive a direction to alter the parameters of our model in.  

It doesn't work when the error function is _discrete_ as this doesn't allow us enough distinction to improve the parameters, i.e. there are many ways to achieve an error rating of 2; the difference between rolling down a hill, and taking the stairs one at a time. When you are present on the step that you are currently on, then it is hard to quantify which direction you should go in order to descend to the next one down. Whereas with a slope, small variations in our positions will create small variations in our __continuous__ error function. In reality, our error function must be differentiable in order to perform gradient descent optimisation. 

In conclusion, to apply gradient descent the error function should:  
1. Be differentiable
2. Be continuous

To achieve this, we will need to move from discrete predictions, to continuous ones. We can do this by altering our activation function from the steps function to the smoother sigmoid function, we then get a probability of the class being a given state as our prediction; we can later round this to our needs to get a class prediction output. 