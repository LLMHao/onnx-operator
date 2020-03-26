# Local Response normalization

- AlexNet首次提出

## 作用
在深度学习训练时提高准确度

## LRN原理
仿造生物学上活跃的神经元对相邻神经元的抑制现象（侧抑制（个人认为类似于生物学上的顶端优势））。通过对局部神经元的活动创建竞争机制，使得其中响应较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

## 公式
output = input / (bias + alpha * sqr_sum) ** beta
https://img-blog.csdn.net/20170713162906129?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMjE1ODU3ODU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast
[https://blog.csdn.net/yangdashi888/article/details/77918311](https://blog.csdn.net/yangdashi888/article/details/77918311)

## 属性（Atrributes）
alpha

## 输入
x:T
输入的数据tensor；图像的shape为(N x C x H x W ),N是batch size， C是通道数。如果是s其他类型的则输入维度表示为[DATA_BATCH,DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE...]

## 输出 Outputs
y:T 
输出tensor，与输入tensor的的维度和类型一致。
限制：
输入和输出必须为浮点数


## numpy模拟
```
alpha = 0.0001
beta = 0.75
bias = 1.0
nsize = 3
node = onnx.helper.make_node(
    'LRN',
    inputs=['x'],
    outputs=['y'],
    size=3
)
x = np.random.randn(5, 5, 5, 5).astype(np.float32)
square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
for n, c, h, w in np.ndindex(x.shape):
    square_sum[n, c, h, w] = sum(x[n,
                                   max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                   h,
                                   w] ** 2)
y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
expect(node, inputs=[x], outputs=[y],
       name='test_lrn_default')
```

## 注意
- tensorflow 对shape的解释为N*H*W*C 与标准不一致
- 公式中 alpha / nsize这个缩放因子不一致

[TensorFlow 参考https://blog.csdn.net/banana1006034246/article/details/75204013](https://blog.csdn.net/banana1006034246/article/details/75204013)

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMzQyMzQyODEsLTIyMTUwNDkwOF19
-->