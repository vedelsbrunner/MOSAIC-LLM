from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import requests
import json
load_dotenv()


def get_prompt_txt(prompt_path) -> str:
    prompt_txt = ""
    with open(
            prompt_path, "r"
    ) as f:
        prompt_txt = f.read()  # has

    return prompt_txt


def create_query_optimization(model, output_parser):
    query_optimization_prompt = get_prompt_txt("prompts/query_optimization.txt")

    optimization_prompt_template = ChatPromptTemplate.from_messages(
        [("human", query_optimization_prompt)],
    )
    return optimization_prompt_template | model | output_parser


def create_query_summarization(model, output_parser):
    res_summary_prompt = get_prompt_txt("prompts/result_summarization.txt")

    res_summary_prompt_template = ChatPromptTemplate.from_messages(
        [("human", res_summary_prompt)],
    )
    return res_summary_prompt_template | model | output_parser


def query_mosaic(query):
    url = f"https://qnode.eu/ows/mosaic/service/search?q={query}"
    query_result = ''
    try:
        response = requests.get(url)
        response.raise_for_status()
        query_result = response.json()
        # print(json.dumps(json_response, indent=4))
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

    return query_result


def main():
    model = ChatMistralAI(model="open-mixtral-8x7b", temperature=0.7)
    output_parser = StrOutputParser()

    query_optimizer = create_query_optimization(model, output_parser)
    summary_optimizer = create_query_summarization(model, output_parser)

    original_query = "What is the purpose of life?"
    print("Original query: " + original_query)

    optimized_query = json.loads(query_optimizer.invoke({"q": original_query}))

    print("Optimized query: ", optimized_query['clarified_query'])
    mosaic_result = query_mosaic(optimized_query['clarified_query'])
    print(json.dumps(mosaic_result, indent=4))



if __name__ == '__main__':
    main()
