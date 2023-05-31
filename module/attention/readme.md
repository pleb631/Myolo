# 测试
测试环境:笔记本gtx1050
|module|param(k)|flops(k)|speed(ms)|year|
|---|---|---|---|---|
|SE|32|133|0.19|2017|
|CBAM|32|185|0.64|2018|
|ECA|4个|102|0.18|2019|
|CA|25|675|0.65|2021|
|SOCA|66|65|3.3|2019|
|SimAM|0|-|0.15|2020|
|S2_MLPv2|2e3|2e5|0.71|2021|
|NAM|1|401|0.24|2021|
|CCNet|328|6.4e3|1.37|2018|
|GAM|6.5e3|1.28e6|4|2021|
|ShuffleAtten|-|50|0.34|2021|
|Biformer|1e3|2.06e5|1|2023|
|EffectiveSE|262|262|0.14|2019|
|MHSA|788|1.55e5|0.42|2017|
|PolarizedAtten|527|7.7e4|0.8|2021|
|SGE|-|101|0.23|2019|
|TripletAtten|300个|1.4e3|0.94|2020|
|GC|
|BAM|
# 详解

- SE  
    [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)  
    ![2023-05-31-09-11-36](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-09-11-36.png)  
    关键词:通道  
- CBAM  
    [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf)  
    ![2023-05-31-09-17-04](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-09-17-04.png)  
    关键词:通道.空间  
- ECA  
    [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)  
    ![2023-05-31-09-37-32](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-09-37-32.png)  
    关键词:通道
- CA  
    [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)  
    ![2023-05-31-13-22-42](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-13-22-42.png)
    关键词:通道,空间
- SOCA
    [Second-order Attention Network for Single Image Super-Resolution](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf)
    关键词：超分辨率,空间  
- SimAM  
    [SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks](http://proceedings.mlr.press/v139/yang21o/yang21o.pdf)  
    关键词:无参,能量  
- S2_MLPv2  
    [S^2-MLPV2: IMPROVED SPATIAL-SHIFT MLP ARCHITECTURE FOR VISION](https://arxiv.org/pdf/2108.01072.pdf)  
    ![2023-05-31-15-28-14](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-15-28-14.png)  
    关键词:空间,偏移  
- NAM  
    [NAM: Normalization-based Attention Module](https://arxiv.org/pdf/2111.12419.pdf)  
    ![2023-05-31-15-35-52](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-15-35-52.png)  
    关键词:通道，归一化  
- CCNet
    [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/pdf/1811.11721.pdf)  
    ![2023-05-31-15-41-16](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-15-41-16.png)  
    关键词:十字注意力  
- GAM  
    [Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions](https://arxiv.org/pdf/2112.05561v1.pdf)  
    ![2023-05-31-16-31-38](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-16-31-38.png)
    关键词:通道,空间  
- ShuffleAtten  
    [SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS](https://arxiv.org/pdf/2102.00240.pdf)  
    ![2023-05-31-16-53-57](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-16-53-57.png)  
    关键词:分组,空间,通道  
- Biformer  
    [BiFormer: Vision Transformer with Bi-Level Routing Attention](https://arxiv.org/abs/2303.08810)  
    ![2023-05-31-17-39-48](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-17-39-48.png)  
    关键词:VIT  
- EffectiveSE  
    [CenterMask : Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/pdf/1911.06667.pdf)  
- MHSA  
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PolarizedAtten
    [Polarized Self-Attention: Towards High-quality Pixel-wise Regression](https://arxiv.org/abs/2107.00782)
    ![2023-05-31-18-01-32](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-18-01-32.png)
- SGE
    [Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/pdf/1905.09646.pdf)
- TripletAtten
    [Rotate to Attend: Convolutional Triplet Attention Module](https://arxiv.org/pdf/2010.03045.pdf)
    ![2023-05-31-18-15-22](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-31-18-15-22.png)
