# 安装wsl2
```
PowerShell 或 Windows 命令提示符中输入

$ wsl --install
正在安装: 虚拟机平台
已安装 虚拟机平台。
正在安装: 适用于 Linux 的 Windows 子系统
已安装 适用于 Linux 的 Windows 子系统。
正在安装: Ubuntu
已安装 Ubuntu。
请求的操作成功。直到重新启动系统前更改将不会生效。

重启电脑

Ubuntu 已安装。
正在启动 Ubuntu...
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username:

输入unix用户名，两次输入密码，安装完成

Installation successful!
```

# 安装 Ubuntu-22.04 LTS
```
$ wsl --list -o
以下是可安装的有效分发的列表。
使用 'wsl.exe --install <Distro>' 安装。

NAME                                   FRIENDLY NAME
Ubuntu                                 Ubuntu
Debian                                 Debian GNU/Linux
kali-linux                             Kali Linux Rolling
Ubuntu-18.04                           Ubuntu 18.04 LTS
Ubuntu-20.04                           Ubuntu 20.04 LTS
Ubuntu-22.04                           Ubuntu 22.04 LTS
OracleLinux_7_9                        Oracle Linux 7.9
OracleLinux_8_7                        Oracle Linux 8.7
OracleLinux_9_1                        Oracle Linux 9.1
openSUSE-Leap-15.5                     openSUSE Leap 15.5
SUSE-Linux-Enterprise-Server-15-SP4    SUSE Linux Enterprise Server 15 SP4
SUSE-Linux-Enterprise-15-SP5           SUSE Linux Enterprise 15 SP5
openSUSE-Tumbleweed                    openSUSE Tumbleweed
```

```
$ wsl --install Ubuntu-22.04
正在安装: Ubuntu 22.04 LTS
已安装 Ubuntu 22.04 LTS。
正在启动 Ubuntu 22.04 LTS...
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username:

输入unix用户名，两次输入密码，安装完成

操作成功完成。
Installation successful!
```

# 常用操作
1. 导入导出
```
a) 查看所有wsl
$ wsl -l --all -v
  NAME            STATE           VERSION
* Ubuntu          Stopped         2
  Ubuntu-22.04    Stopped         2

b) 导出WSL(需要先建好目标文件夹 d:\WSL）
$ wsl --export Ubuntu-22.04 d:\WSL\ubuntu22.04.tar
正在导出，这可能需要几分钟时间。
操作成功完成。

c) 注销WSL
$ wsl --unregister Ubuntu-22.04
正在注销。
操作成功完成。

d) 导入WSL（一定要写–version 2！）
$ wsl --import Ubuntu-22.04 d:\WSL\Ubuntu-22.04 d:\WSL\ubuntu22.04.tar --version 2
正在导入，这可能需要几分钟时间。
操作成功完成。

e) Ubuntu修改默认登陆用户(我觉得用root挺好的[狗头]) ubuntu2204.exe config --default-user Username

f) 原地导入 WSL
$ wsl --import-in-place Ubuntu D:\WSL\Ubuntu-22.04\ext4.vhdx

g) 原地导入权限问题 
Error code: Wsl/Service/CreateInstance/MountVhd/E_ACCESSDENIED 
给Users完全控制权限
https://blog.csdn.net/weixin_37178320/article/details/128720616
```
2. 控制内存使用量
```
编辑 C:\Users<UserName>.wslconfig
[wsl2]
memory=30GB
```
[更多高级控制功能](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

3. 使用 Windows Terminal
这玩意比命令提示符和powershell好用多了，安利

# 安装 Anaconda
[Anaconda](https://www.anaconda.com/download/success)

```
bash Anaconda3-2024.02-1-Linux-x86_64.sh

这样写对于第一次使用linux环境的是不是粗糙了一点..
```

以下版本经过测试

Anaconda3-2024.02-1-Linux-x86_64.sh


# 更新Ubuntu&安装开发工具包
```
sudo apt-get update
sudo apt-get upgrade
sudo apt install build-essential -y
```

选装
```
sudo apt  install nvtop -y
```

# 之后的操作
[跳转链接](./setup#安装依赖)
