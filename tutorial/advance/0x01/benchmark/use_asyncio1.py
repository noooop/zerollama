import time
import asyncio
import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from use_sync.client import Client as BaseClient
from use_asyncio.server import ZeroServerProcess, Z_MethodZeroServer, ZeroServerResponseOk


class Client(BaseClient):
    def test(self):
        data = {"method": "test"}
        return self.query(data)


class Server(Z_MethodZeroServer):
    async def z_test(self, req):
        await asyncio.sleep(0.1)
        rep = ZeroServerResponseOk(msg={})
        await self.zero_send(req, rep)


def client(x):
    t1 = time.time()
    client = Client("tcp://localhost:9527")
    client.test()
    t2 = time.time()
    return x, t1, t2


if __name__ == '__main__':
    import numpy as np
    from multiprocessing import Pool

    h = ZeroServerProcess(Server, {"do_register": False, "port": 9527})
    h.start()

    nn = 1000
    ht = 100

    for n in [1, 2, 4, 8, 16, 32]:
        p = Pool(n)

        b = [o for o in p.imap(client, range(nn))]
        b = b[ht:-ht]
        latency = np.mean([x[2]-x[1] for x in b])
        s = min([x[1] for x in b])
        e = max([x[2] for x in b])
        qps = (nn-ht*2)/(e-s)

        print(f"{n} processes. Latency: {latency:0.4f}, QPS: {qps:0.4f}")

    h.terminate()
    h.join()


