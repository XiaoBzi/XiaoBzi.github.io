---
layout:     post
title:      "py小白工作总结，持续更新……"
description: "常用命令总结+可行解决方法"
excerpt: "航拍+8D重庆+赛国朋克"
date:    2024-01-04
author:     "甜甜圈"
image: "img/fj1.jpg"
published: true 
tags:
    - ubuntu 
URL: "/worktipsleecode/"
categories: [ "我的工作" ]    
---
## 命令
#### 1. 数组(常用命令)  
```html
nums=[1,1,2,3]
del nums[i]
nums.count(1)
```
#### 2.轮转数组  
```html
nums[:]=nums[-k:] + nums[:-k]
```
#### 3. 摩尔投票，同归于尽，找比一半人数还要多的票
摩尔pk投票算法(同归于尽算法)是一种用于找出数组中出现次数最多的元素的线性时间算法。它的基本思想是通过不断消除不同的元素对来找出最终的候选元素。
算法步骤如下：初始化候选元素(candidate)和计数器(count)：首先将候选元素设为数组的第一个元素，计数器设为1。
遍历数组：从第二个元素开始遍历数组。
更新候选元素和计数器：对于每个遍历到的元素，如果计数器为0，则将候选元素更新为当前元素，计数器设为1。如果当前元素与候选元素相同，则计数器加1；如果不同，则计数器减1。
返回候选元素：遍历完成后，候选元素即为出现次数最多的元素。

Boyer-Moore投票算法的核心在于，对于任意一对不同的元素，如果它们两两消除，最终剩下的元素仍然是出现次数最多的元素。因此，通过遍历数组，将不同的元素两两消除，最终剩下的就是出现次数最多的元素。

形象化描述：“同归于尽消杀法” ：由于多数超过50%, 比如100个数，那么多数至少51个，剩下少数是49个。

第一个到来的士兵，直接插上自己阵营的旗帜占领这块高地，此时领主 winner 就是这个阵营的人，现存兵力 count = 1。

如果新来的士兵和前一个士兵是同一阵营，则集合起来占领高地，领主不变，winner 依然是当前这个士兵所属阵营，现存兵力 count++；

如果新来到的士兵不是同一阵营，则前方阵营派一个士兵和领主阵营的一个士兵同归于尽。 此时前方阵营兵力count --。（即使双方都死光，这块高地的旗帜 winner 依然不变，因为已经没有活着的士兵可以去换上自己的新旗帜，但是不用考虑这种情况，因为双方死光，下一个占领的直接就是winner，就是第一种情况。）

当下一个士兵到来，发现前方阵营已经没有兵力，新士兵就成了领主，winner 变成这个士兵所属阵营的旗帜，现存兵力 count ++。

就这样各路军阀一直以这种以一敌一同归于尽的方式厮杀下去，直到少数阵营都死光，那么最后剩下的几个必然属于多数阵营，winner 就是多数阵营。（多数阵营 51个，少数阵营只有49个，剩下的2个就是多数阵营的人）

#### 4. 股票随时可买可退是贪心算法  
```html
    def maxProfit(prices):
    total_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            total_profit += prices[i] - prices[i - 1]
    return total_profit
```
#### 5. 股票只能买一次，是最大减最小  
```html
def maxProfit(prices):
    if not prices:
        return 0
    
    min_price = float('inf')  # 初始化为正无穷大
    max_profit = 0  # 初始化最大利润为0
    
    for price in prices:
        if price < min_price:
            min_price = price  # 更新最低价格
        elif price - min_price > max_profit:
            max_profit = price - min_price  # 更新最大利润
    
    return max_profit
```
#### 6. AnimateDiff 有两个版本！
2个版本的animatediff模块对比
![加载中……](/img/tipcomfy/gen.png)
gen2节点寻找方法
![加载中……](/img/tipcomfy/gen2.png)
gen2,流行工作流常用，连接如下：
![加载中……](/img/tipcomfy/gennew.png)
#### 7. IPAadapter 里面,如果clip vision 老是报模型和权重文件结构不符，要重新下载下面的clip vision
```html
https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors
```
#### 8. hugo这个模板写post日期的时候要20xx-xx-xx,要加0，不然不显示！而且能重复。
#### 9. 命令行指定cuda号，跑python代码
```html
CUDA_VISIBLE_DEVICES=X python your_script.py
```
#### 10. 解决深度学习网络中模块，一个在cpu,一个在cuda xxx
```html
device = next(self.parameters()).device
        
        # 确保输入张量 x 和 y 在同一设备上
        x = x.to(device)
        y = y.to(device)

#创建新张量时也改为：
conv_weight_hd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=conv_weight.device, dtype=conv_weight.dtype)

```
#### 11. yolov11jiaonang,目标检测onnx,推理代码，大框套小框处理方法
![加载中……](/img/tipcomfy/iouandconf.png)
#### 12. The command to install mmcv:
```html
pip install -U openmim
mim install mmcv
```
#### 13. pytorch 环境配置 一般都装python 3.9:
```html
python 3.9 可以适配的pytorch  
 conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch
```
#### 14. 视频.音频免费提取网站，处理各种格式转换，免费:
```html
https://www.zamzar.com/

```
#### 15. yolov目标检测（v8,v11）如何加入负样本背景图:
```html
简单来说，负样本加入数（即neg_num）是为了“在不同的训练轮数/策略下更改负样本加入的数量”
例如：
在出现误检模型的基础上，快速训练30轮压误检时：每轮都加负样本
在重新完整训练300/500轮时：随机不加/加点负样本，以保证模型正负样本都充分学习

而在上述情况下，推荐的neg_num设置为：
继续训练30轮————正数2
重新训练300轮————负数负1 （负样本很多时推荐为-1）
                        
原文链接：https://blog.csdn.net/qq_41848886/article/details/134289698

```
#### 16. name：edm2 扩散模型的图片预处理和训练comment :
```html
1.python dataset_tool.py convert --source=/share_data/PRDATA/ljl/DDPM/dataset --dest=datasets/img512.zip --resolution=512x512 --transform=center-crop-dhariwal 
2.python dataset_tool.py encode --source=datasets/img512.zip --dest=datasets/img512-sd.zip 
3.CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 train_edm2.py --outdir=training-runs/00000-edm2-img512-xxs --data=datasets/img512-sd.zip --preset=edm2-img512-xxs

```
#### 17. name：edm2 使用模型权重产生图片comment seed是张数编号:
```html
 CUDA_VISIBLE_DEVICES=3 python generate_images.py --preset=edm2-img512-xs-fid --outdir=out --subdirs --seeds=0-9
```
