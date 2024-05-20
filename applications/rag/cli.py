
import click
from zerollama.tasks.chat.cli import chat
from zerollama.tasks.retriever.cli import retriever
from zerollama.tasks.reranker.cli import reranker


@click.command()
def run():
    from zerollama.microservices.standalone.server import start
    start()


@click.group()
def server():
    pass


server.add_command(run)


@click.command()
@click.argument('config_filename', type=click.Path(exists=True))
def deploy(config_filename):
    from zerollama.microservices.workflow.rag.deploy import Deploy
    Deploy(config_filename)()


@click.group()
def main():
    pass


main.add_command(chat)
main.add_command(retriever)
main.add_command(retriever)
main.add_command(reranker)
main.add_command(deploy)


if __name__ == '__main__':
    main()
