---
layout:     post
title:      "AIGC帖子总结模块功能篇"
description: "实用模块功能+使用方法"
excerpt: "航拍+8D重庆+赛国朋克"
date:    2024-09-06
author:     "甜甜圈"
image: "img/fj1.jpg"
published: true 
tags:
    - ubuntu 
URL: "/comfyuimokuai/"
categories: [ "我的工作" ]    
---
## Comfyui手册
#### 1. 组节点的启用（高亮）和停用
![加载中……](/img/tipcomfy/启用节点.png)  
#### 2. ControlNet 连接位置
连在在clip text 后面，后面退出到ksample采样器正负提示词。
controlnet的预处理器通过图片蓝线传controlnet，例如姿态识别
![加载中……](/img/tipcomfy/controlnet.png) 
#### 3. ControlNet 预处理器 简介
ControlNet 是一个很强的插件，提供了很多种图片的控制方式，有的可以控制画面的结构，有的可以控制人物的姿势，还有的可以控制图片的画风，这对于提高AI绘画的质量特别有用。接下来就演示几种热门常用的控制方式
#### * 1.OpenPose（姿态控制预处理器）
姿态控制预处理器可以根据提供的图像将人物的骨骼脸部手部的姿态展示处理，通过这个预处理器可以很好的控制出图人物的姿态
#### * 2.Depth（深度预处理器）
深度预处理器可以将图片的空间的远近以黑白的形式展示出来，白近黑远，当我们上传一张图片通过OpenPose识别到手的位置，但骨骼图并不能描述手在身前还是身后的时候，那个深度预处理器就可以提现出作用了，当然还可以运用在一些建筑、室内等情况
#### * 3.LineArt（线条预处理器）
线条预处理器可以将图片用线条的形式描绘出来，可以很好的控制图片的细节
#### * 4.HED Soft-Edge（模糊线条预处理器）
模糊线条预处理器与线条预处理器类型也是用线条描绘图片，但仅大概描绘轮廓，更利于出图的随机性。
![加载中……](/img/tipcomfy/controlnet预处理器.jpg)
#### 4. 找资源（对于不能翻墙的我）  
```html
https://hf-mirror.com/
```
#### 5. Comfyui加噪声，案例在github readme 
```html
https://github.com/BlenderNeko/ComfyUI_Noise.git
```
![加载中……](/img/tipcomfy/加噪.png)
加噪都作用：
模型鲁棒性测试：在测试生成模型时，通过添加噪声来评估模型的鲁棒性和稳定性。
数据增强：在潜空间中生成多样化的训练数据，增强模型的泛化能力。

#### 6. comfyui抠图模块 更好的遮罩 LayerMask
![加载中……](/img/tipcomfy/抠图.png)
#### 7. comfyui滤镜模块 Cube 滤镜文件
![加载中……](/img/tipcomfy/滤镜.png)
#### 8.comfyui亮度调节模块 自动调光
![加载中……](/img/tipcomfy/调光.png)
使用方法：
![加载中……](/img/tipcomfy/调光工作流.png)
#### 9. comfyui图片上色模块
![加载中……](/img/tipcomfy/上色.png)
使用方法：
![加载中……](/img/tipcomfy/上色工作流.jpg)
#### 10. Image Blur 节点 羽化 模糊
该节点用于对输入的图像进行模糊处理，以改变图像的视觉效果或者减少图像中的细节，通常用于创建柔和或者抽象化的视觉效果。
blur_radius → 输入模糊的高斯半径
sigma → 该值越小，模糊的像素越接近中心
![加载中……](/img/tipcomfy/羽化.png)
使用方法：
![加载中……](/img/tipcomfy/羽化工作流.jpg)
#### 11. Image Sharpen 节点 锐化 清晰
该节点用于增强图像的清晰度和细节，通常用于提升图像的视觉效果和边缘锐化。
sharpen_radius → 表示锐化的半径
sigma → 该值越小，锐化的像素越接近中心像素
alpha → 锐化的强度
使用方法：
![加载中……](/img/tipcomfy/锐化工作流.jpg)
#### 12. Image Quantize 节点 灰度图 简单色彩
该节点用于将输入的图像进行量化处理，即将图像中的颜色数目减少到较少的色彩级别。
colors → 表示量化后图像包含的颜色数量（颜色数量最小为1，最大位256）
dither → 添加抖动效果，使图像在量化后更加平滑，会更好
使用方法：
![加载中……](/img/tipcomfy/减色工作流.png)
#### 13. Image Blend 节点 花了
该节点用于将两幅图像混合在一起，生成一个结合了两幅图像特征的图像。
![加载中……](/img/tipcomfy/混图.png)
#### 14. 文本生成图像不需要向潜在空间输入任何东西，使用空latent
![加载中……](/img/tipcomfy/空latent.png)
#### 15. 输入参考图的要使用vae编码，latent缩放大小再输入ksample
![加载中……](/img/tipcomfy/vaeencoder.png)
#### 16. Latent加vae进行局部重绘。
![加载中……](/img/tipcomfy/局部重绘.jpg)
#### 17. 批量生成图片，latent复制批次，次数次
![加载中……](/img/tipcomfy/批次输出.jpg)
#### 18. 串联重设latent批次省内存
![加载中……](/img/tipcomfy/节省内存批次.png)
#### 19.获取某批次中的某一张图片，如此标号，从批次获取latent
![加载中……](/img/tipcomfy/选择性输出.jpg)
#### 20.Comfyui脸部修复 这个约等于ksample 直接用
![加载中……](/img/tipcomfy/脸部修复工作流.png)
寻找该节点方法：
![加载中……](/img/tipcomfy/找脸部修复.png)
找其他组件：
![加载中……](/img/tipcomfy/找其他修复.png)
#### 21.VAE Decode 在ksample后面解析为图片。tile 是省内存版
tile_size -> 用来设定"块"的大小，进行多块处理
Tips：这个节点特别适合处理高分辨率图像，因为将图像分成小块进行处理，然后在处理完所有块后将它们组合成完整图像，可以避免内存溢出，并且提高处理效率。
例如，1024*1024的图像可以通过设置1024，来分成2块进行解码，也可以设置成512，分成4块进行解码。
![加载中……](/img/tipcomfy/vaetile.png)
#### 22.图生图工作流
![加载中……](/img/tipcomfy/图生图工作流.jpg)
#### 23.获
