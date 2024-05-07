import time
import asyncio
import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from use_asyncio.client import Client as BaseClient
from use_asyncio.server import ZeroServerProcess, Z_MethodZeroServer, ZeroServerResponseOk


class Client(BaseClient):
    async def test(self):
        data = {"method": "test"}
        return await self.query(data)


class Server(Z_MethodZeroServer):
    async def z_test(self, req):
        await asyncio.sleep(0.1)
        rep = ZeroServerResponseOk(msg={})
        await self.zero_send(req, rep)


async def client(x):
    t1 = time.time()
    client = Client("tcp://localhost:9527")
    await client.test()
    t2 = time.time()
    return x, t1, t2


if __name__ == '__main__':
    import numpy as np

    h = ZeroServerProcess(Server, {"do_register": False, "port": 9527})
    h.start()

    nn = 1000
    ht = 100

    # https://death.andgravity.com/limit-concurrency

    async def map_unordered(func, iterable, *, limit):
        try:
            aws = map(func, iterable)
        except TypeError:
            aws = (func(x) async for x in iterable)

        async for task in limit_concurrency(aws, limit):
            yield await task


    async def limit_concurrency(aws, limit):
        try:
            aws = aiter(aws)
            is_async = True
        except TypeError:
            aws = iter(aws)
            is_async = False

        aws_ended = False
        pending = set()

        while pending or not aws_ended:
            while len(pending) < limit and not aws_ended:
                try:
                    aw = await anext(aws) if is_async else next(aws)
                except StopAsyncIteration if is_async else StopIteration:
                    aws_ended = True
                else:
                    pending.add(asyncio.ensure_future(aw))

            if not pending:
                return

            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            while done:
                yield done.pop()


    async def run():
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            b = []
            async for result in map_unordered(client, range(nn), limit=n):
                b.append(result)

            b = b[ht:-ht]
            latency = np.mean([x[2]-x[1] for x in b])
            s = min([x[1] for x in b])
            e = max([x[2] for x in b])
            qps = (nn-ht*2)/(e-s)

            print(f"{n} processes. Latency: {latency:0.4f}, QPS: {qps:0.4f}")

    asyncio.run(run())

    h.terminate()
    h.join()