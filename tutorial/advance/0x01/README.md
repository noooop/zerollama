# 为什么异步编程可以提高吞吐
异步的关键在于，你在等待一个任务完成的同时，可以执行其他任务

想象Z_MethodZeroServer是一个办事大厅，每个Z_Method是一组服务窗口，由引导员负责将客户引导到具体的服务窗口，服务结束后由引导员将客户送走，
只要客户到了服务窗口，引导员就可以等待用户完成服务，同时去引导其他客户，这时只需要一个引导员就可以完成所有客户的引导任务。

比如引导员在每个客户身上花的时间（包括切换客户的时间）不超过0.1ms，就能达成一万QPS。相反，如果有一个请求需要引导员租塞一秒，其他客户都会卡住。

具体的就需要将CPU、GPU密集的计算放在子进程执行，io密集的让客户端等待到io响应（POLLIN， POLLOUT..），其他资源需要等待的，让客户端等待在信号量（Semaphore）上。

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

# 2. 使用gevent
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

gevent简单有效，虽然做不到对异步完全无感，但对代码逻辑不用做大的改动，使用客户端门槛比较低，上手比较快

# 使用 asyncio
async 就像一坨屎一样，有一个函数使用了 async def，调用它的函数都得用async def，还要asyncio.run运行，项目里要迁就着异步框架。
这个函数再也不是简单的python函数，而沾了屎的函数。叹息