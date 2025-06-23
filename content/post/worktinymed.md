---
layout:     post
title:      "轻量医学多模态大模型浅测试"
description: "Med-MoE,stage3"
excerpt: "航拍+8D重庆+赛国朋克"
date:    2025-04-13
author:     "甜甜圈"
image: "img/fj1.jpg"
published: true 
tags:
    - ubuntu 
URL: "/worktinymed/"
categories: [ "我的工作" ]    
---
## Med-MoE,轻量医学多模态大模型，使用stage3模型权重 全部大约20G
今天调通了一个医学图像多模态大模型，但是发现它的效果很差，不知道是不是我所有模型都是使用的最小的模型导致的。
以下记录调通模型的一步步，方便以后使用
代码地址：
```html
原始：https://github.com/jiangsongtao/Med-MoE
模型权重位置：https://huggingface.co/JsST/TinyMed/tree/main/Stage3/MoE-Tinymed-phi2-llava
```
使用stage3 就是多模态文图问答模型 
还需要下载clip模型，调用图像塔
```html
https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main
```
需要修改stage3的图像塔为本地位置 "mm_image_tower": "/share_data/PRDATA/ljl/Med-MoE-main/clip",
运行 deepspeed predict.py 测试代码
问题1：deepspeed未初始化
解决：添加以下代码，可能还要安装一个包，pip安装不了，conda可以 
```html
 install -c conda-forge mpi4py
 ```
 监测deepspeed是否初始化成功代码：
 ```html
def main():
     # 禁用 PyTorch 的默认初始化（如果需要）
    disable_torch_init()

    # 初始化 DeepSpeed 分布式通信
    import deepspeed
    deepspeed.init_distributed(dist_backend='nccl')  # 使用 NCCL 后端（适用于 GPU）

    # 检查通信后端是否初始化成功
    from deepspeed.comm import comm
    if comm.cdb is None:
        raise RuntimeError("DeepSpeed communication backend is not initialized.")
    else:
        print("DeepSpeed communication backend initialized successfully.")

    # 加载图像、输入文本、模型路径等参数
```
还要好多是调不出来图像塔对问题
需要在moellava/model/builder.py里面将from deepspeed.moe.layer import MoE打开
我将会把代码上传到自己的GitHub里面。

