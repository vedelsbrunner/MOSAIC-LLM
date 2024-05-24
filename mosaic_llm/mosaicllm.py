import dataclasses
from typing import ClassVar
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
import ast
import json
import requests

load_dotenv()


@dataclasses.dataclass
class MosaicLLM:
    model_name: str = "open-mixtral-8x7b"
    temperature: int = 0.7
    root: str = "../"

    QUERY_OPTIMIZER_PATH: ClassVar = "prompts/query_optimization.txt"
    RESULT_SUMMARIZATION_PATH: ClassVar = "prompts/result_summarization.txt"

    # HELPERS
    @staticmethod
    def get_prompt_txt(prompt_path) -> str:
        prompt_txt = ""
        with open(prompt_path, "r") as f:
            prompt_txt = f.read()  # has
        return prompt_txt

    @staticmethod
    def query_mosaic(query):
        url = f"https://qnode.eu/ows/mosaic/service/search?q={query}?&index=demo-simplewiki&lang=eng&limit=5"
        query_result = ""
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

    @staticmethod
    def extract_textsnippet_from_mosaic_response(response):
        text_snippets = []

        for result in response["results"]:
            for item in result["demo-simplewiki"]:
                try:
                    text_snippets.append(item["textSnippet"])
                except:
                    print("Missing text, skipping..")

        return text_snippets

    def get_prompt_template(self, path):
        query_optimization_prompt = MosaicLLM.get_prompt_txt(path)

        optimization_prompt_template = ChatPromptTemplate.from_messages(
            [("human", query_optimization_prompt)],
        )
        return optimization_prompt_template

    def __post_init__(self):
        load_dotenv()
        self.model = ChatMistralAI(model=self.model_name, temperature=self.temperature)

        self.query_optimizer_prompt = self.get_prompt_template(
            f"{self.root}{MosaicLLM.QUERY_OPTIMIZER_PATH}"
        )
        self.query_optimizer_chain = (
            self.query_optimizer_prompt | self.model | StrOutputParser()
        )

        self.result_summarizer_prompt = self.get_prompt_template(
            f"{self.root}{MosaicLLM.RESULT_SUMMARIZATION_PATH}"
        )
        self.result_summarizer_chain = (
            self.result_summarizer_prompt | self.model | StrOutputParser()
        )

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(1))
    def optimize_query(self, query: str) -> dict:
        params: dict = {"q": query}
        raw_asnwer = self.query_optimizer_chain.invoke(params)

        llm_answer = ast.literal_eval(raw_asnwer)
        dict_ = json.loads(llm_answer) if isinstance(llm_answer, str) else llm_answer

        prompt_str: str = self.query_optimizer_prompt.invoke(params).to_string()
        return prompt_str, dict_


    @retry(stop=stop_after_attempt(10), wait=wait_fixed(1))
    def summarize_results(self, query: str, result_snippet: list) -> str:
        result_snippet_str: str = "- " + "-".join(result_snippet)
        params = {"q": query, "snippet_lst": result_snippet_str}
        answer = self.result_summarizer_chain.invoke(params)
        prompt_str: str = self.result_summarizer_prompt.invoke(params).to_string()
        return prompt_str, answer

    def search_and_summarize(self, query: str):
        response = MosaicLLM.query_mosaic(query)
        snippet_lst = MosaicLLM.extract_textsnippet_from_mosaic_response(response)
        prompt_str, summary = self.summarize_results(query, snippet_lst)
        return prompt_str, summary

    def run(self, query: str) -> dict:
        result = {}
        # optimize
        result["query_optimizer_prompt"], optimized_query_json = self.optimize_query(
            query
        )
        result = optimized_query_json

        # QUERY MOSAIC and summarize
        result["prompt_query_summary"], result["summary_query"] = (
            self.search_and_summarize(query)
        )
        result["prompt_clarified_query_summary"], result["summary_clarified_query"] = (
            self.search_and_summarize(optimized_query_json["clarified_query"])
        )

        all_snippets = []
        queries = []
        for suggested_query in optimized_query_json["subqueries"]:
            queries.append(suggested_query)
            response = MosaicLLM.query_mosaic(suggested_query)
            all_snippets.extend(
                MosaicLLM.extract_textsnippet_from_mosaic_response(response)
            )
        result["prompt_subqueries"], result["summary_subqueries"] = (
            self.summarize_results(", ".join(queries), all_snippets)
        )
        return result
