import numpy as np

def my_file_function(list, multiplier):
    new_list = [x * multiplier for x in list]
    return(new_list)

def my_file_function_2(list, multiplier):
    array = np.array(list)
    new_list = array * multiplier
    return(new_list)