

def server():
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        message = socket.recv()
        for i in range(10):
            socket.send_string(f"{i:010d}", zmq.SNDMORE)
        socket.send_string("done!")


def client():
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    for request in range(10):
        print("Client sending request %s …" % request)
        socket.send(b"Hello")

        part1 = socket.recv()
        print(part1)

        while socket.getsockopt(zmq.RCVMORE):
            part = socket.recv()
            print(part)


if __name__ == '__main__':
    from multiprocessing import Process

    s = Process(target=server)
    c = Process(target=client)

    s.start()
    c.start()
    c.join()
    s.terminate()
