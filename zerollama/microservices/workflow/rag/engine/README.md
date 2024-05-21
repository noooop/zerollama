# 向量数据库

## demo
```
第一个窗口
$ python -m zerollama.workflow.vector_database.engine.server
Use Vector Database backend:
zerollama.workflow.vector_database.use_bruteforce:BruteForceVectorDatabase
ZeroNameServer: InMemoryNameServer running! port: 9527
BruteForceVectorDatabase: is running! port: 51717

第二个窗口
$ python -m zerollama.workflow.vector_database.engine.client
================================================================================
Wait VectorDatabase available
[ServerInfo(name='0eea0b5a04e53d2a1095b94545415402', host='localhost', port=51717, protocol='vector_database')]
================================================================================
VectorDatabase support_methods
state='ok' msg={'name': 'ZeroVectorDatabaseEngine', 'support_methods': ['info', 'support_methods', 'top_k']}
state='ok' msg={'vector database class': 'BruteForceVectorDatabase'}

......


# 第一个窗口 Control-C 关闭 
ZeroNameServer clean_up!
ZeroVectorDatabaseEngine clean_up!
```