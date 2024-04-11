

def server():
    import zmq
    import json

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        message, *payload = socket.recv_multipart()
        message = json.loads(message)

        response = json.dumps({
            "state": "ok",
            "msg": {
                "n_payload": len(payload)
            },
            "meta": message["meta"]
        }).encode('utf8')

        #time.sleep(1)
        socket.send_multipart([response] + payload)


def client():
    import time
    import zmq
    import numpy as np
    import json

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    for request in range(10):
        tt = []
        for i in range(100):
            t = time.time()
            tenser = np.random.rand(3, 224, 224).astype(np.float32)
            buffer = tenser.tobytes()

            message = json.dumps({
                "meta": [
                    {
                        "dtype": "float32",
                        "shape": [3, 224, 224],
                    }
                ]
            }).encode('utf8')
            payload = [buffer]
            #print("Client sending request %s …" % request)
            socket.send_multipart([message] + payload)

            message, *payload = socket.recv_multipart()
            #print("Client received reply %s" % request)
            message = json.loads(message)

            for buffer, meta in zip(payload, message["meta"]):
                tenser = np.frombuffer(buffer, dtype=meta["dtype"]).reshape(meta["shape"])
                #print(tenser.shape)
            tt.append(time.time() - t)
        print(request, np.mean(tt) * 1000, "ms")


if __name__ == '__main__':
    from multiprocessing import Process

    s = Process(target=server)
    c = Process(target=client)

    s.start()
    c.start()
    c.join()
    s.terminate()
