
import click
from zerollama.tasks.chat.cli import chat
from zerollama.tasks.retriever.cli import retriever


@click.command()
def run():
    from zerollama.microservices.standalone.server import start
    start()


@click.command()
@click.argument('config_filename', type=click.Path(exists=True))
def init(config_filename):
    from zerollama.microservices.standalone.server import setup, run

    server = setup()
    run(server, waiting=False)

    def deploy():
        from zerollama.microservices.standalone.deploy import Deploy
        Deploy(config_filename)()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as executor:
        f = executor.submit(deploy)
        f.result()

    for h in server:
        h.wait()


@click.group()
def server():
    pass


server.add_command(run)
server.add_command(init)


@click.command()
@click.argument('config_filename', type=click.Path(exists=True))
def deploy(config_filename):
    from zerollama.microservices.standalone.deploy import Deploy
    Deploy(config_filename)()


@click.group()
def main():
    pass


main.add_command(chat)
main.add_command(retriever)

main.add_command(deploy)
main.add_command(server)


if __name__ == '__main__':
    main()
