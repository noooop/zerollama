
import re
from string import Template
from zerollama.microservices.workflow.modular_rag.rq_rag.prompt_template import template


def rewriter_messages(original_query, rewritten_queries=None):
    rewritten_queries = rewritten_queries or []

    content = Template(template.QueryRewriterTemplate).safe_substitute(original_query=original_query, rewritten_queries=rewritten_queries)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


def judger_messages(original_query, retrieved_evidences, reference_output):
    content = Template(template.QueryJudgerTemplate).safe_substitute(original_query=original_query,
                                                                     retrieved_evidences=retrieved_evidences,
                                                                     reference_output=reference_output)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


def queryGenerator_messages(contexts, original_query):
    content = Template(template.DecomposeGeneratorTemplate).safe_substitute(contexts=contexts, original_query=original_query)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


def unambiguous_short_messages(original_query, search_results, original_answers):
    content = Template(template.UnambiguousGeneratorTemplateShort).safe_substitute(original_query=original_query,
                                          search_results=search_results,
                                          original_answers=original_answers)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


def unambiguous_long_messages(ambiguous_question, unambiguous_questions_with_answers, original_answer):
    content = Template(template.UnambiguousGeneratorTemplateShort).safe_substitute(ambiguous_question=ambiguous_question,
                                          unambiguous_questions_with_answers=unambiguous_questions_with_answers,
                                          original_answer=original_answer)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


def decomposer_messages(contexts, original_query):
    content = Template(template.DecomposeGeneratorTemplate).safe_substitute(contexts=contexts, original_query=original_query)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


def multiturn_retriever_rewriter_messages(conversation_history, current_query):
    content = Template(template.MultiTurnGeneratorTemplate).safe_substitute(conversation_history=conversation_history, current_query=current_query)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


def parser_multiturn_retriever_rewriter_results(response):
    # assume the response can be splitted by \n, and every element is a dict
    retrieval_necessity = re.search(r'Retrieval Necessity:\s*(\w+)', response)
    queries = re.search(r'Query For Search Engine:\s*([\s\S]+)', response)

    # Output the results
    retrieval_necessity = retrieval_necessity.group(1) if retrieval_necessity else None
    queries = queries.group(1).split("\n") if queries else None

    if retrieval_necessity in ["yes", "Yes", "YES"]:
        retrieval_necessity = True

    if (retrieval_necessity is None) or (queries is None):
        return False, []

    return retrieval_necessity, queries


def qa_retriever_rewriter_messages(current_query):
    content = Template(template.QAGeneratorTemplate).safe_substitute(current_query=current_query)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


if __name__ == '__main__':
    from zerollama.tasks.chat.engine.client import ChatClient
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "Qwen/Qwen1.5-14B-Chat-AWQ"

    client = ChatClient()
    client.wait_service_available(model_name)

    test_rewriter = True
    test_judger = False
    test_decomposer = False
    test_unambiguous = False
    test_multiturn_retriever_rewriter = False
    test_qa_retriever_rewriter = False

    if test_rewriter:
        messages = rewriter_messages("When are the 2024 NBA playoffs scheduled to begin?")
        response = client.chat(model_name, messages)
        print(response.msg.content)

    if test_judger:
        messages = judger_messages(original_query="What kind of university is the school where Rey Ramsey was educated an instance of?",
                                   retrieved_evidences=["When are the 2024 NBA playoffs scheduled to begin?"],
                                   reference_output="")
        response = client.chat(model_name, messages)
        print(response.msg.content)

    if test_decomposer:
        messages = decomposer_messages(contexts="",
                                       original_query="What kind of university is the school where Rey Ramsey was educated an instance of?")
        response = client.chat(model_name, messages)
        print(response.msg.content)

    if test_multiturn_retriever_rewriter:
        messages = multiturn_retriever_rewriter_messages(conversation_history="",
                                                          current_query="Hello, today is a good day!")
        response = client.chat(model_name, messages)
        retrieval_necessity, queries = parser_multiturn_retriever_rewriter_results(response.msg.content)
        print(retrieval_necessity)
        print(queries)

        messages = multiturn_retriever_rewriter_messages(conversation_history="",
                                                          current_query="where is the capital of france?")
        response = client.chat(model_name, messages)
        retrieval_necessity, queries = parser_multiturn_retriever_rewriter_results(response.msg.content)
        print(retrieval_necessity)
        print(queries)

        messages = multiturn_retriever_rewriter_messages(conversation_history="",
                                                          current_query="What kind of university is the school where Rey Ramsey was educated an instance of?")
        response = client.chat(model_name, messages)
        retrieval_necessity, queries = parser_multiturn_retriever_rewriter_results(response.msg.content)
        print(retrieval_necessity)
        print(queries)

    if test_qa_retriever_rewriter:
        messages = qa_retriever_rewriter_messages(current_query="Hello, today is a good day!")
        response = client.chat(model_name, messages)
        retrieval_necessity, queries = parser_multiturn_retriever_rewriter_results(response.msg.content)
        print(retrieval_necessity)
        print(queries)

        messages = qa_retriever_rewriter_messages(current_query="where is the capital of france?")
        response = client.chat(model_name, messages)
        retrieval_necessity, queries = parser_multiturn_retriever_rewriter_results(response.msg.content)
        print(retrieval_necessity)
        print(queries)

        messages = qa_retriever_rewriter_messages(current_query="What kind of university is the school where Rey Ramsey was educated an instance of?")
        response = client.chat(model_name, messages)
        retrieval_necessity, queries = parser_multiturn_retriever_rewriter_results(response.msg.content)
        print(retrieval_necessity)
        print(queries)


