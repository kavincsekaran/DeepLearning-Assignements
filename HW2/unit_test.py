import numpy as np
a=np.array(range(10))
b=np.array(np.arange(0,5,0.5))
print(a-b)
if(np.all(a-b>0.5)):
    print(a-b)