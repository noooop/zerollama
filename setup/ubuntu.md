# 安装ubuntu

参考 [Win10下Linux双系统的安装教程](https://zhuanlan.zhihu.com/p/362092786)

**注意**
1. ext4分区不容易后期调整大小，考虑到大语言模型动辄几个G，至少给ubuntu准备100G的空间
2. 安装时选择使用NVIDIA专有驱动程序，如果选择默认安装开源驱动程序 Nouveau 之后还要卸载重装一遍


## ubuntu 发行版
以下版本经过测试

ubuntu-22.04.4-desktop-amd64.iso

## NVIDIA显卡驱动
以下版本经过测试

Driver Version: 535.171.04 

软件和更新|附加驱动 使用NVIDIA driver metapackage 来自nvidia-driver-535 (专有，tested)

# 之后的操作参考 wsl2
[跳转链接](./wsl2.md#%E5%AE%89%E8%A3%85-anaconda)