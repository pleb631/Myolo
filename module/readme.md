# Explanation  

- **CoordConv**  
    [PP-YOLO: An Effective and Efficient Implementation of Object Detector](https://arxiv.org/pdf/2007.12099.pdf)
    ![2023-05-24-10-19-21](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-24-10-19-21.png)  
    与传统卷积相比，CoordConv就是在输入的feature map后面增加了两个通道，一个表示x坐标，一个表示y坐标，后面就是与正常卷积过程一样了。CoordConv可以类比残差连接  
- **SAConv**  
    [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf)  
    ![2023-05-24-13-21-04](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-24-13-21-04.png)  
    作者将ResNet中所有的3×3的卷积层都转换为SAC，以实现在不同空洞率的情况下卷积操作的软切换。图中红色的锁表示权重是相同的，只是训练过程可能有差异。全局上下文模块和SENet有以下两个不同之处：1.只有一个卷积层，没有任何的非线性层；2.输出被加回到主干中，而不是和输入相乘。通过实验发现，在SAC之前添加全局上下文模块，会对检测性能产生好的影响  
- **ODConv**  
    [OMNI-DIMENSIONAL DYNAMIC CONVOLUTION](https://openreview.net/pdf?id=DmpCfq6Mg39)  
    ![2023-05-24-14-30-29](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-24-14-30-29.png)  
    关键词:动态卷积  
    对于卷积核： (1) 在空间位置的卷积参数（每个滤波器）赋予权重；(2) 为每个卷积滤波器输入通道分配不同的注意标量；(3)将不同的注意标量分配给卷积滤波器；(4)给整个卷积核分配一个注意标量。原则上，这四种类型的注意是互补的，并以位置、通道、滤波器和核的顺序逐步将它们乘以卷积核，使得卷积操作为不同的输入x的所有空间位置、所有输入通道、所有过滤器和所有内核，为捕获丰富的上下文线索提供了性能保证  
- **PConv**  
    [Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks](https://arxiv.org/pdf/2303.03667.pdf)  
    ![2023-05-24-14-48-36](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-24-14-48-36.png)  
    关键词:轻量化  
    它只需在输入通道的一部分上应用常规Conv进行空间特征提取，并保持其余通道不变。对于连续或规则的内存访问，将第一个或最后一个连续的通道分组视为整个特征图的代表进行计算。之后需要加上1x1卷积或DW卷积进行全局计算  
- **DCNv2**  
    [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)  
    ![2023-05-24-15-20-26](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-24-15-20-26.png)  
    关键词:可变形卷积  
- **DCNv3**  
    [github](https://github.com/OpenGVLab/InternImage)  
- **CARAFE**  
    [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188)  
    关键词:上采样,动态卷积  
- **CAM**  
    [Context Aaugmentation And Feature Refinement Network For tiny Object Detection](https://openreview.net/pdf?id=q2ZaVU6bEsT)  
    关键词：扩大感受野,空洞卷积  
