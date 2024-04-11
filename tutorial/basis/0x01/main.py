

def server():
    import time
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        message = socket.recv()
        print("Server received request: %s" % message)

        time.sleep(1)
        socket.send(b"World")


def client():
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    for request in range(10):
        print("Client sending request %s …" % request)
        socket.send(b"Hello")

        message = socket.recv()
        print("Client received reply %s [ %s ]" % (request, message))


if __name__ == '__main__':
    from multiprocessing import Process

    s = Process(target=server)
    c = Process(target=client)

    s.start()
    c.start()
    c.join()
    s.terminate()
