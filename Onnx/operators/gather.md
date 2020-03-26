### 属性
axis：int (默认为0)
gather执行所在的维度

## onnx gather与numpy take 功能和概念等价

[参考 numpy.take  ](https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html)

numpy.take()

```python
import numpy as np
a = [4,3,5,7,6,8]
indices=[0,1,4]
np.take(a,indices)
```
indices 是标量和向量的区别示例，可以通过unsqueeze将标量转换为一维向量
```
import numpy as np
a=np.array([2,3,4,6])
indices =  3
b = np.take(a,indices)
print(b)# 类型为int32 类型
indices2 = [3]
c = np.take(a,indices2)
print(c)#类型为一维的ndarray
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExNzkyMjg3MDhdfQ==
-->