import re


def text_window_parser(file, window_size=4):
    sentence_splitter = lambda text: re.findall("[^,.;。？！]+[,.;。？！]?", text)
    book_name = file.stem.split("-")[0]
    text = file.read_text(encoding="utf-8")
    lines = text.split('\n')

    sentence_list = []
    for i, line in enumerate(lines):
        if len(line) == 0:
            continue
        text_splits = sentence_splitter(line)
        for line in text_splits:
            sentence_list.append({
                "name": book_name,
                "text": line
            })

    nodes = []
    for i, sentence in enumerate(sentence_list):
        window_sentence = sentence_list[
                          max(0, i - window_size): min(i + window_size, len(sentence_list))
                          ]
        nodes.append({
            "name": book_name,
            "text": " ".join([sentence["text"] for sentence in window_sentence]),
            "window": window_sentence
        })
    return lines, sentence_list, nodes


if __name__ == '__main__':
    from pathlib import Path

    rag_path = Path.home() / ".zerollama/rag/documents"

    for dir in rag_path.glob("*"):
        print(dir)
        for file in dir.glob("*.txt"):
            lines, sentence_list, nodes = text_window_parser(file)
            book_name = file.stem.split("-")[0]
            print(book_name)

            print(f"一共有{len(lines)}个段落")
            print(f"一共有{len(sentence_list)}个句子, 最大长度为{max(len(line['text']) for line in sentence_list)}")
            print(f"处理完一共有{len(nodes)}个节点")
    print()