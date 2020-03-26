# 绝对值 

等价于:  numpy.abs

## Abs
inputs
x:T  输入tensor

outputs
y:T 输出tensor

```
node = onnx.helper.make_node(
    'Abs',
    inputs=['x'],
    outputs=['y'],
)
```
```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def abs(input):  # type: (np.ndarray) -> np.ndarray
    return np.abs(input)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = abs(x)

expect(node, inputs=[x], outputs=[y],
       name='test_abs')
```



 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNjg1OTAyNjZdfQ==
-->