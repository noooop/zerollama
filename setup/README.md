# Windows 11
[详细配置](./windows.md)

# Windows 11 + wsl2
[详细配置](./wsl2.md)

# Ubuntu
[详细配置](./ubuntu.md)

# 试试conda虚拟环境
环境管理允许用户方便安装不同版本的python环境，并在不同环境之间快速地切换。

## 自动配置
```
# ubuntu&wsl2
$ conda env create -f environment_linux.yml

# windows
$ conda env create -f environment_windows.yml
```
区别是 ubuntu&wsl2 使用pip安装pytorch， windows使用conda安装pytorch

网络问题多试几次，找找代理，比如[TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/),自己克服一下吧，叹气

下面的配置都已经一键安装好了

## conda 常用命令
```
# 创建虚拟环境
$ conda create -n zerollama_v0.5 python=3.11 anaconda

# 查看虚拟环境
$ conda env list 

# 激活虚拟环境
$ conda activate zerollama_v0.5

# 安装依赖, 只会安装在这个虚拟环境里
# pip install .....

# 停用虚拟环境
$ conda deactivate

# 删除虚拟环境
conda remove -n zerollama_v0.5 --all
```

# clone 工程代码
```
$ git clone https://github.com/noooop/zerollama.git
```

# 切换到工作目录
项目默认的工作目录是项目根目录
```
$ cd zerollama
```

# 配置文件
[跳转链接](https://github.com/noooop/zerollama/tree/main/zerollama/core/config)
