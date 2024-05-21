# TL;DR
经过一些实验，我决定 zerollama 项目依然使用 gevent 协程，而不使用 asyncio。

(十年前 asyncio 是一坨屎，现在还是一坨屎)

> Beautiful is better than ugly.
> 
> Explicit is better than implicit.
> 
这两句显然会冲突

gevent:
- Implicit & Beautiful
- 优势 对同步代码逻辑不用做大的改动，函数（看上去）是同步的，使用起来门槛也比较低。
- 劣势 monkey patch太魔法，尤其是patch socket和 event 有时会导致多进程代码卡死。
  没有显式区分异步调用函数，不注意也会导致阻塞主进程。这些问题暂时都可以通过调试和对slow log分析解决。
  asyncio 也只是好一点点，也需要大量调试。异步编程就是很头疼。

asyncio:
- Explicit & Ugly
- 优势 asyncio 架构和抽象层次都非常好，复杂系统确实需要这些。无论是官方库还是第三方库，生态要好一些。
  await 关键字明确在协程之间转移执行控制的点，没有使用魔法，做了什么导致出错也比较好调试。
- 劣势 async 就像一坨屎一样，有一个函数使用了 async def，调用它的函数都得用async def。
  async 函数还要 asyncio.run 运行，项目里要迁就着异步框架，无论是写代码还是用代码，门槛很高。
  使用 asyncio 就像用一门新语言，时刻提醒你正在吃屎。
  新手甚至老手哪怕用一个最简单的功能都会被其复杂性震惊到，这也太复杂了。
- 劣势2 asyncio 内置的 event loop 性能比较差，不如 gevent。 当然也可以用第三方 event loop，这也太复杂了。
- 劣势3 asyncio 真的是一个很低级的库，甚至没有 worker pool，不能p.imap_unordered，得自己控制并发，这也太复杂了。
- 劣势4 asyncio 跟 windows 的兼容性有很大问题。zerollama 想做成一个跨平台框架，而不是只能运行在linux上

或许之后会有兼容性问题，或者是因为项目太复杂，导致 zerollama 需要用 asyncio 重写，暂时 gevent 用起来还不错。

# 为什么协程可以提高吞吐
协程的关键在于，你在等待一个任务完成的同时，可以执行其他任务。

想象Server是一个办事大厅，每个方法是一组服务窗口，由引导员负责将客户引导到具体的服务窗口，服务结束后由引导员将客户送走，
只要客户到了服务窗口，引导员就可以等待用户完成服务，同时去引导其他客户，这时只需要一个引导员就可以完成所有客户的引导任务。

比如引导员在每个客户身上花的时间（包括切换客户的时间）不超过0.1ms，就能达成一万QPS。相反，如果有一个请求需要引导员租塞，其他客户都会卡住。

具体的就需要将CPU、GPU密集的计算放在子线程子进程（ThreadPoolExecutor、ProcessPoolExecutor）执行，io密集的让客户端等待到io响应（POLLIN， POLLOUT..），其他资源需要等待的，让客户端等待在信号量（Semaphore）上。

协程的关键在于，不要阻塞主进程。


# 1. 同步服务端
```
$ python use_sync1.py 
Server is running! port: 9527
1 processes Latency: 0.1016, QPS: 9.8339
2 processes Latency: 0.2013, QPS: 9.9222
4 processes Latency: 0.4025, QPS: 9.9000
8 processes Latency: 0.8084, QPS: 9.8098
16 processes Latency: 1.6095, QPS: 9.7580
32 processes Latency: 3.2177, QPS: 9.5742
Server clean_up!
```

可以看到同步服务端，毕竟只有一个进程工作，增加客户端数量，延迟成倍增加，吞吐不变甚至小幅下降

# 2. 使用 gevent
如果服务端使用gevent，只需要简单修改，延迟保持不变或者小幅下降的情况下，QPS跟客户端数量线性增加

```
$ python use_gevent1.py 
Server is running! port: 9527
1 processes. Latency: 0.1088, QPS: 9.1827
2 processes. Latency: 0.1086, QPS: 18.3717
4 processes. Latency: 0.1087, QPS: 36.6658
8 processes. Latency: 0.1213, QPS: 65.1704
16 processes. Latency: 0.1216, QPS: 129.7434
32 processes. Latency: 0.1298, QPS: 234.6923
Server clean_up!
```

如果客户端也使用gevent，客户端只用一个进程就可以做512甚至更高的并发，QPS跟客户端数量线性增加，直到用完服务器POOL_SIZE

```
POOL_SIZE = 64
$ python use_gevent2.py 
Server is running! port: 9527
1 processes. Latency: 0.1081, QPS: 9.1960
2 processes. Latency: 0.1085, QPS: 18.3588
4 processes. Latency: 0.1081, QPS: 36.6323
8 processes. Latency: 0.1079, QPS: 72.5699
16 processes. Latency: 0.1079, QPS: 143.4769
32 processes. Latency: 0.1085, QPS: 281.1915
64 processes. Latency: 0.1085, QPS: 531.2040
128 processes. Latency: 0.2442, QPS: 450.4373
256 processes. Latency: 0.4258, QPS: 482.1710
512 processes. Latency: 0.6617, QPS: 464.0581
Server clean_up!

POOL_SIZE = 128
$ python use_gevent2.py 
Server is running! port: 9527
1 processes. Latency: 0.1090, QPS: 9.1315
2 processes. Latency: 0.1087, QPS: 18.3250
4 processes. Latency: 0.1077, QPS: 36.7102
8 processes. Latency: 0.1078, QPS: 72.6145
16 processes. Latency: 0.1079, QPS: 143.5335
32 processes. Latency: 0.1081, QPS: 280.4522
64 processes. Latency: 0.1074, QPS: 536.2712
128 processes. Latency: 0.1095, QPS: 965.2285
256 processes. Latency: 0.1976, QPS: 882.8355
512 processes. Latency: 0.2933, QPS: 904.9475
Server clean_up!
```

gevent简单有效，虽然做不到对异步完全无感，但对代码逻辑不用做大的改动，使用客户端门槛比较低，上手比较快。

# 使用 asyncio
async 就像一坨屎一样，有一个函数使用了 async def，调用它的函数都得用async def，还要asyncio.run运行，项目里要迁就着异步框架。

```
# 同步客户端
$ python use_asyncio1.py 
Server is running! port: 9527
1 processes. Latency: 0.1096, QPS: 9.1160
2 processes. Latency: 0.1123, QPS: 17.7522
4 processes. Latency: 0.1098, QPS: 36.3231
8 processes. Latency: 0.1151, QPS: 68.2356
16 processes. Latency: 0.1109, QPS: 140.4323
32 processes. Latency: 0.1226, QPS: 238.9560
```

为什么没有 Server clean_up! 
因为 [Windows does not have signals](https://stackoverflow.com/questions/45987985/asyncio-loops-add-signal-handler-in-windows). 
我甚至不知道该抱怨 Windows 兼容性，还是抱怨 asyncio 兼容性。
先这样吧，也不想再修了

没有 worker pool 可以控制并发，我也不太想自己写一个

```
# gevent 客户端 
$ python use_asyncio2.py 
Server is running! port: 9527
1 processes. Latency: 0.1121, QPS: 8.9104
2 processes. Latency: 0.1135, QPS: 17.5394
4 processes. Latency: 0.1130, QPS: 34.9997
8 processes. Latency: 0.1146, QPS: 68.2981
16 processes. Latency: 0.1153, QPS: 133.1972
32 processes. Latency: 0.1091, QPS: 271.9962
64 processes. Latency: 0.1116, QPS: 481.6314
128 processes. Latency: 0.1073, QPS: 987.7698
256 processes. Latency: 0.1168, QPS: 1373.7313
512 processes. Latency: 0.1847, QPS: 1532.6284
```


客户端还是得 worker pool
[参考](https://death.andgravity.com/limit-concurrency)
反正 asyncio 是一门新语言

```
# asyncio 客户端 
$ python use_asyncio3.py 
Server is running! port: 9527
1 processes. Latency: 0.1121, QPS: 8.9104
2 processes. Latency: 0.1135, QPS: 17.5394
4 processes. Latency: 0.1130, QPS: 34.9997
8 processes. Latency: 0.1146, QPS: 68.2981
16 processes. Latency: 0.1153, QPS: 133.1972
32 processes. Latency: 0.1091, QPS: 271.9962
64 processes. Latency: 0.1116, QPS: 481.6314
128 processes. Latency: 0.1073, QPS: 987.7698
256 processes. Latency: 0.1168, QPS: 1373.7313
512 processes. Latency: 0.1847, QPS: 1532.6284
```

done！ asyncio 再见