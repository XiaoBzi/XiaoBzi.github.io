---
layout:     post
title:      "工作经验的小小总结，持续工作，持续更新……"
description: "常用命令总结+可行解决方法"
excerpt: "航拍+8D重庆+赛国朋克"
date:    2024-09-02
author:     "甜甜圈"
image: "img/fj1.jpg"
published: true 
tags:
    - ubuntu 
URL: "/worktips/"
categories: [ "我的工作" ]    
---
## 命令
#### 1. 清华源镜像  
```html
pip install XXXX -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
#### 2. github镜像源  
```html
https://kkgithub.com/
```
#### 3. comfyui的node装不起  （界面爆红）
点击名字超链接，进入GitHub下载zip,然后解压到XXXX-node里面，如果还是不行，要进入下好的文件夹

```html
删除一切版本号，删除torch那三个，numpy,opencv
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python COMFUI/main.py
```
#### 4. 找资源（对于不能翻墙的我）  
```html
https://hf-mirror.com/
```
#### 5. git上传  
```html
cd 项目
hugo
hugo serve
first
 git init
 git remote remove origin
 git remote add origin https://github.com/XXX项目地址
 git pull （抓取）
 then
 git add .(本地)
 git commit -m "每次不同英文"
 git push --set-upstream origin master
```