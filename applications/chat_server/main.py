

if __name__ == '__main__':
    import subprocess
    s = subprocess.Popen("python -m applications.chat_server.server", shell=True)
    h = subprocess.Popen("python -m applications.chat_server.http_gateway", shell=True)

    s.wait()

