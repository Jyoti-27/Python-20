#!/usr/bin/env python
# coding: utf-8

# Creating 1D Array.

# In[2]:


import numpy as np


# In[2]:


arr=np.array([1,2,3])
print(arr)


# In[4]:


print(arr.ndim)


# In[5]:


print(arr.shape)


# In[6]:


print(arr.dtype)


# In[7]:


print(arr.itemsize)


# In[9]:


arr=np.array([[1,2,3],[4,5,6]])
print(arr)


# In[10]:


print(arr.ndim) # ndim tells the dimension of array


# In[11]:


print(arr.shape)  # shape tells the no. of rows and columns in an array


# In[12]:


print(arr.dtype)


# In[13]:


print(arr.itemsize)


# In[14]:


arr=np.array([[[1,2,3],[4,5,6]],[[10,20,30],[40,50,60]]])
print(arr)


# In[15]:


arr=np.array([[[1,2,3],[4,5,6]],[[10,20,30],[40,50,60]], [[1,2,3],[4,5,6]]])
print(arr)


# In[16]:


print(arr.ndim)


# In[17]:


print(arr.shape) # 3= 3 dimensional, 2= 2 elements in each, 3= 3 columns in each


# In[18]:


arr=np.array([[[1.5,2.5,3.5]], [[10.0,20.5,30]]])
print(arr)


# In[19]:


print(arr.ndim)


# In[20]:


print(arr.shape)


# Shape tells us the number of elements in each dimension of an array arr[axis0, axis1, axis2]

# In[21]:


arr=np.array([[[1,2,3,1], [4,5,6,1], [7,8,9,1]], [[10,20,30,1], [40,50,60,1], [7,8,9,1]]])
print(arr)


# In[22]:


print(arr.ndim)


# In[23]:


print(arr.shape)


# Create 2 matrices and take their transpose and add them.

# In[25]:


matrix=np.array([[1,2,3], [4,5,6]])


# In[26]:


print(matrix)


# In[33]:


matrix1=np.array([[1,2,3], [4,5,6]])
matrix2=np.array([[10,20,30], [40,50,60]])


# In[34]:


print(matrix1)
print(matrix2)


# In[35]:


print(matrix1.T)


# In[36]:


print(matrix1.shape)


# In[37]:


print(matrix1.T.shape)


# In[38]:


matrix3 = matrix1 + matrix2


# In[39]:


print(matrix3)


# In[40]:


matrix3 = matrix1 - matrix2


# In[41]:


print(matrix3)


# In[42]:


vector=np.array([10,11,12])


# In[43]:


print(vector)


# In[44]:


print(vector.T)


# In[45]:


arr2=np.array([[[1,2,3,4,5],[4,5,6,7]],[[8,9,10,11],[10,20,30,12]],[[40,50,60,70],[70,80,90,100]]])
print(arr2)


# In[46]:


print(arr2.shape)


# In[47]:


print(arr2.dtype)


# In[48]:


print(arr2.ndim)


# In[49]:


print(arr2.size)


# In[50]:


print(arr2.itemsize)


# In[19]:


array2=np.array([[1,2,3],[3,4,5],[7,8,0]])


# In[52]:


print(array2)


# In[53]:


print(array2.mean)


# In[54]:


print(np.mean(array2))  # Default mean gives the mean of all elements and mean of all elementsin 2d array


# In[55]:


print(np.mean(array2, axis=0))   # It finds the mean of each column 


# In[56]:


print(np.mean(array2, axis=1))    # It finds the mean of each row


# In[60]:


array2=np.array([[1,2,3],[3,4,5],[7,8,0]])   # Created a 2d array
print("mean columnwise")
print(np.mean(array2, axis=0))   # mean columnwise
print("std columnwise")
print(np.std(array2, axis=0))   # Standard deviation
print("var columnwise")
print(np.var(array2, axis=0))   # Variance
print("max columnwise")
print(np.max(array2, axis=0))   # maximum of each column
print("min columnwise")
print(np.min(array2, axis=0))   # minimum of each column
print("sum columnwise")
print(np.sum(array2, axis=0))   # adds elements of matrix columnwise
print("prod columnwise")
print(np.prod(array2, axis=0))  # product of elements in each column


# In[61]:


array2.sum(axis=0)


# In[20]:


np.sum(array2,axis=0)


# In[21]:


np.any(array2,axis=0)


# In[22]:


np.any(array2)


# In[24]:


print(array2)


# In[25]:


np.any(array2)


# In[27]:


array2.any(axis=0)


# In[28]:


array2.all(axis=0)


# In[29]:


array2=np.array([[1,2,3], [3,4,5], [7,9,0]])
print(array2)


# In[30]:


array2.all(axis=0)


# In[31]:


array2.all()


# In[32]:


array2=np.array([[1,2,3], [3,4,5], [7,9,10]])
print(array2)


# In[35]:


array2.any(axis=0)   #


# In[33]:


array2=np.array([[0,2,3], [0,4,5], [0,9,0]])
print(array2)


# In[34]:


array2.any(axis=0)   # if a column has all zeros , it returns false


# Array Indexing Slicing Updating

# In[5]:


arr=np.array[1,2,4,5,6]
print(arr)
print(arr[0], arr[2])  # Access elements of 1d array
print(arr[2:4])  # Access element 2 and 3 ends with 1 less
print(arr[2:5])
print(arr[:5])   # Starts from 0 to 4
print(arr[2:])   # Starts  from 2 till end
arr[2]=200       # array Updation
print(arr)   


# Accessing Slicing and Updation 2D array

# In[16]:


arr=np.array([[1,2,4,5,6], [1.2,1.4,2.4,5.6,6.7]])
print(arr)
print(arr[1][1], arr[0][2])  # Access elements of 2d array
print(arr[0], arr[1])
#print(arr[slicing of row,slicing of column])
print(arr[1:2,  2:4])  # Access element 2 and 3 ends with 1 less
print(arr[:, 1:2])
print(arr[1:3,  3:])
print(arr[1:, 2:])
arr[1,4]=500  # Array Updation
print(arr)
#print(arr[2:5])
#print(arr[:5])   # Starts from 0 to 4
#print(arr[2:])   # Starts  from 2 till end
#arr[2]=200       # array Updation
#print(arr)   


# In[40]:


arr1=np.array([[1,2,3,5,6],[6,7,8,9,7],[12,14,45,66,7],[44,55,67,55,7]])
print(arr1)
print(arr1[0:3, 0:3])


# In[2]:


# Create array from list
import numpy as np
list1 = [1, 2, 3, 4, 5]
arr = np.array(list1)   # Send list1 as a parameter to an array
print(arr)


# In[5]:


# Using arange function of numpy
import numpy as np
arr = np.arange(10,100)   # Arrange (start, stop, step)
print(arr)


# In[6]:


# Using arange function of numpy
import numpy as np
arr = np.arange(4,10,2)   # Arrange (start, stop, step)
print(arr)


# In[7]:


# Creating Array with Zeros where is it useful
import numpy as np
zero_arr = np.zeros((5,5))   # Arrange (start, stop, step)
print(zero_arr)


# In[8]:


# Creating Array with ones where is it useful
import numpy as np
ones_arr = np.ones((4,3))   # Arrange (start, stop, step)
print(ones_arr)


# In[9]:


# Array of given size filled with value 2.5
full_array = np.full((4,3), 2.5)   # Arrange (start, stop, step)
print(full_array)


# How to create 2D array using range?

# In[10]:


arr=np.arange(0,10)
print(arr)


# In[11]:


arr=np.arange(0,10).reshape(2,5)
print(arr)


# In[17]:


arr=np.arange(0,12).reshape(2,3)
print(arr)


# In[16]:


arr=np.arange(0,12).reshape(2,3,2)
print(arr)


# In[19]:


arr=np.arange(0,12).reshape(2,3,2).ravel()
print(arr)


# In[20]:


# Creating array using linspace
import numpy as np
vector = np.linspace(0, 20, 5)
print(vector)
# Observe the difference between linspace and arange


# In[21]:


# Creating array using linspace
import numpy as np
vector = np.linspace(0, 20, 3)   # linspace(start, stop, number of values)
print(vector)
# Observe the difference between linspace and arange


# In[22]:


# Creating array using linspace
import numpy as np
vector = np.linspace(0, 20, 8)   # linspace(start, stop, number of values)
print(vector)
# Observe the difference between linspace and arange


# In[24]:


# A 3x3 Array of random values uniformly distributed between 0 and 1
ran_arr = np.random.random((3,3))
print(ran_arr)


# In[25]:


# A 3x4 array of random integers in the interval(90,10)
np.random.randint(0, 10, (3, 4))


# In[4]:


arr21=np.array([[[1,2,3,5],[4,5,6]]])
print(arr21)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




