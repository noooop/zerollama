

if __name__ == '__main__':
    import subprocess
    s = subprocess.Popen("python server.py", shell=True)
    h = subprocess.Popen("python http_gateway.py", shell=True)

    s.wait()

