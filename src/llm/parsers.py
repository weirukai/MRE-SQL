import json
import re
import logging
from ast import literal_eval
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.exceptions import OutputParserException


class PythonListOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing Python lists."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract Python list content from markdown.

        Args:
            output (str): The output string containing Python list.

        Returns:
            Any: The parsed Python list.
        """
        logging.debug(f"Parsing output with PythonListOutputParser: {output}")
        if "```python" in output:
            output = output.split("```python")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return eval(output)  # Note: Using eval is potentially unsafe, consider using ast.literal_eval if possible.


class FilterColumnOutput(BaseModel):
    """Model for filter column output."""
    chain_of_thought_reasoning: str = Field(
        description="One line explanation of why or why not the column information is relevant to the question and the hint.")
    is_column_information_relevant: str = Field(description="Yes or No")


class SelectTablesOutputParser(BaseOutputParser):
    """Parses select tables outputs embedded in markdown code blocks containing JSON."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract JSON content from markdown.

        Args:
            output (str): The output string containing JSON.

        Returns:
            Any: The parsed JSON content.
        """
        logging.debug(f"Parsing output with SelectTablesOutputParser: {output}")
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        output = output.replace("\n", " ").replace("\t", " ")
        return json.loads(output)


class ColumnSelectionOutput(BaseModel):
    """Model for column selection output."""
    table_columns: Dict[str, Tuple[str, List[str]]] = Field(
        description="A mapping of table and column names to a tuple containing the reason for the column's selection and a list of keywords for data lookup. If no keywords are required, an empty list is provided.")


class GenerateCandidateOutput(BaseModel):
    """Model for SQL generation output."""
    chain_of_thought_reasoning: str = Field(
        description="Your thought process on how you arrived at the final SQL query.")
    SQL: str = Field(description="The generated SQL query in a single string.")


class GenerateCandidateFinetunedMarkDownParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "```sql" in output:
            output = output.split("```sql")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return {"SQL": output}


class ReviseOutput(BaseModel):
    """Model for SQL revision output."""
    chain_of_thought_reasoning: str = Field(
        description="Your thought process on how you arrived at the final SQL query.")
    revised_SQL: str = Field(description="The revised SQL query in a single string.")


class GenerateCandidateGeminiMarkDownParserCOT(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with RecapOutputParserCOT: {output}")
        plan = ""
        if "<FINAL_ANSWER>" in output and "</FINAL_ANSWER>" in output:
            plan = output.split("<FINAL_ANSWER>")[0]
            output = output.split("<FINAL_ANSWER>")[1].split(
                "</FINAL_ANSWER>"
            )[0]
        query = output.replace("```sql", "").replace("```", "").replace("\n", " ")
        return {"SQL": query, "plan": plan}


class GeminiMarkDownOutputParserCOT(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParserCoT: {output}")
        if "My final answer is:" in output:
            plan, query = output.split("My final answer is:")
        else:
            plan, query = output, output
        if "```sql" in query:
            query = query.split("```sql")[1].split("```")[0]
        query = re.sub(r"^\s+", "", query)
        return {"SQL": query, "plan": plan}


class ReviseGeminiOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with CheckerOutputParser: {output}")
        if "<FINAL_ANSWER>" in output and "</FINAL_ANSWER>" in output:
            output = output.split("<FINAL_ANSWER>")[1].split(
                "</FINAL_ANSWER>"
            )[0]
        if "<FINAL_ANSWER>" in output:
            output = output.split("<FINAL_ANSWER>")[1]
        query = output.replace("```sql", "").replace("```", "").replace("\n", " ")
        return {"refined_sql_query": query}


class ListOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output a list

        Args:
            output (str): A string containing a list.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        try:
            output = literal_eval(output)
        except Exception as e:
            raise OutputParserException(f"Error parsing list: {e}")
        return output


class UnitTestEvaluationOutput(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        scores = []
        if "<answer>" in output and "</answer>" in output:
            output = output.split("<answer>")[1].split(
                "</answer>"
            )[0].strip()
        else:
            return "OutputParserException"
        for line in output.split("\n"):
            if ":" in line:
                try:
                    key, value = line.split(":")
                    if "passed" in value.lower():
                        scores.append(1)
                    else:
                        scores.append(0)
                except Exception as e:
                    import re
                    pattern = r"Candidate Response #\d+: \[([^\]]+)\]"

                    # 使用 re.findall 提取所有匹配的内容
                    matches = re.findall(pattern, output)
                    for match in matches:
                        if "passed" in match.lower():
                            scores.append(1)
                        else:
                            scores.append(0)
                    if len(scores) < 1:
                        return "OutputParserException"
        return {"scores": scores}


class TestCaseGenerationOutput(BaseOutputParser):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<answer>" in output and "</answer>" in output:
            response = output.split("<answer>")[1].split(
                "</answer>"
            )[0]
        else:
            raise OutputParserException(
                "Test generation parse error. Please make sure to include your answer in the format <answer>...</answer>")
        try:
            unit_tests = literal_eval(response)
        except Exception as e:
            try:
                cleaned_response = response.strip()[1:-1]
                tests = cleaned_response.split(",\n")
                unit_tests = [item.strip().strip("'").strip() for item in tests]
            except Exception as e:
                raise OutputParserException(f"Error parsing test case generation: {e}")
        return {"unit_tests": unit_tests}


class TestCaseGenerationJSONOutput(BaseOutputParser):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        try:
            unit_tests = JsonOutputParser().invoke(output)
        except:
            raise OutputParserException(
                "Your test generation answer is not in the correct format. Please make sure to include your answer in the JSON format.")
        # try:
        #     unit_tests = literal_eval(output)
        # except Exception as e:
        #     raise OutputParserException(f"Error parsing test case generation: {e}")
        return {"unit_tests": unit_tests}


class TestCaseGenerationSFTJSONOutput(BaseOutputParser):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")

        try:
            unit_tests = JsonOutputParser().invoke(output)
            unit_tests = [item['test case'] for item in unit_tests]
        except:
            try:
                unit_tests = re.findall(
                    r'''["']test case["']\s*:\s*(["'])(.*?)(?<!\\)\1''',
                    output,
                    re.DOTALL
                )
                if not isinstance(unit_tests[0], str):
                    unit_tests = [i[1] for i in unit_tests]
                if unit_tests:
                    pass
                else:
                    raise OutputParserException(
                        "Your test generation answer is not in the correct format. Please make sure to include your answer in the JSON format."
                    )
            except:
                raise OutputParserException(
                    "Your test generation answer is not in the correct format. Please make sure to include your answer in the JSON format.")
        try:
            unit_tests = literal_eval(str(unit_tests))
        except Exception as e:
            raise OutputParserException(f"Error parsing test case generation: {e}")
        return {"unit_tests": unit_tests}


class BinarySelectionOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.
        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<Answer>" in output and "</Answer>" in output:
            output = output.split("<Answer>")[1].split(
                "</Answer>"
            )[0].strip()
        else:
            return "OutputParserException"
        return output


class BinarySelectionSFTOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.
        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<Answer>" in output and "</Answer>" in output and "<Thinking>" in output and "</Thinking>" in output:
            pass
        else:
            raise OutputParserException(
                "Your binary seleciton answer  is not in the correct format. Please make sure to include your answer in the format <Answer>...</Answer>")
        return output


class ListwiseSelectionOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        # if "<Answer>" in output and "</Answer>" in output:
        #     response = output.split("<Answer>")[1].split(
        #         "</Answer>"
        #     )[0].strip()
        #     response = response.replace("```sql", "").replace("```", "").strip()
        try:
            response = JsonOutputParser().invoke(output)
            response = response['SQL']
        except Exception:
            try:
                response = re.findall(r'SQL"\s*:\s*"((?:[^"\\\\]|\\.)*)"', output, re.DOTALL)[0]
            except Exception:
                return "OutputParserException"
        return response


class GroupwiseSelectionOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<SQL>" in output and "</SQL>" in output:
            response = output.split("<SQL>")[1].split(
                "</SQL>"
            )[0].strip()
            response = response.replace("```sql", "").replace("```", "").strip()
        elif "<SQL>" in output and "</SQL>" not in output:
            response = output.split("<SQL>")[1].strip()
            response = response.replace("```sql", "").replace("```", "").strip()
        else:
            return "OutputParserException"
        return response


class PointwiseSelectionOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
            Parses the output to extract SQL content from markdown.

            Args:
                output (str): The output string containing SQL query.

            Returns:
                Dict[str, str]: A dictionary with the SQL query.
            """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<Answer>" in output and "</Answer>" in output:
            output = output.split("<Answer>")[1].split(
                "</Answer>"
            )[0].strip()
        else:
            return "OutputParserException"
        return output


class QueryRewriteOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "<Answer>" in output and "</Answer>" in output:
            output = output.split("<Answer>")[1].split(
                "</Answer>"
            )[0].strip()
        else:
            raise OutputParserException(
                "Your answer is not in the correct format. Please make sure to include your answer in the format <Answer>...</Answer>")
        return output


def get_parser(parser_name: str) -> BaseOutputParser:
    """
    Returns the appropriate parser based on the provided parser name.

    Args:
        parser_name (str): The name of the parser to retrieve.

    Returns:
        BaseOutputParser: The appropriate parser instance.

    Raises:
        ValueError: If the parser name is invalid.
    """
    parser_configs = {
        "python_list_output_parser": PythonListOutputParser,
        "filter_column": lambda: JsonOutputParser(pydantic_object=FilterColumnOutput),
        "select_tables": lambda: JsonOutputParser(pydantic_object=SelectTablesOutputParser),
        "select_columns": lambda: JsonOutputParser(pydantic_object=ColumnSelectionOutput),
        "generate_candidate": lambda: JsonOutputParser(pydantic_object=GenerateCandidateOutput),
        "generated_candidate_finetuned": GenerateCandidateFinetunedMarkDownParser(),
        "revise": lambda: JsonOutputParser(pydantic_object=ReviseOutput),
        "generate_candidate_gemini_markdown_cot": GenerateCandidateGeminiMarkDownParserCOT(),
        "generate_candidate_gemini_cot": GeminiMarkDownOutputParserCOT(),
        "revise_new": ReviseGeminiOutputParser(),
        "list_output_parser": ListOutputParser(),
        "evaluate": UnitTestEvaluationOutput(),
        "generate_unit_tests": TestCaseGenerationOutput(),
        "generate_unit_tests_two": TestCaseGenerationJSONOutput(),
        "generate_unit_tests_sft": TestCaseGenerationSFTJSONOutput(),
        "binary_selection": BinarySelectionOutputParser(),
        "binary_selection_two": BinarySelectionSFTOutputParser(),
        "listwise_selection": ListwiseSelectionOutputParser(),
        "query_rewrite": QueryRewriteOutputParser(),
        "pointwise_selection": PointwiseSelectionOutputParser(),
        "groupwise_selection": GroupwiseSelectionOutputParser()
    }

    if parser_name not in parser_configs:
        logging.error(f"Invalid parser name: {parser_name}")
        raise ValueError(f"Invalid parser name: {parser_name}")

    logging.info(f"Retrieving parser for: {parser_name}")
    parser = parser_configs[parser_name]() if callable(parser_configs[parser_name]) else parser_configs[parser_name]
    return parser
