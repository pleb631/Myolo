# 测试  
测试环境:笔记本gtx1050
|module|param(M)|flops(G)|speed(ms)|
|---|---|---|---|
|SPPF|0.66|0.26|0.94
|simSPPF|0.66|0.26|0.90
|ASPP|8.9|3.4|8.8|
|RFB|1.3|0.53|2.8|
|SPPCSPC|7.0|2.8|9.4|
# 详解  

- SPPF  
    [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)  
    ![2023-05-30-14-27-35](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-30-14-27-35.png)  
- simSPPF  
    出自yolov6,相对SPPF，激活函数由SILU改为RELU.  
- RFB  
    [Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/abs/1711.07767)  
    ![2023-05-30-15-06-09](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-30-15-06-09.png)  
- SPPFCSPC
    出自yolov7
    ![2023-05-30-15-14-15](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2023-05-30-15-14-15.png)
