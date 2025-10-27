import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
import re
from runner.database_manager import DatabaseManager
from selection_builder import SelectionMethodFactory

from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from runner.task import Task
from tqdm import tqdm
from multiprocessing import Pool
from typing import List
import numpy as np
import yaml
from runner.logger import Logger

import os
from jinja2 import Template

from few_shot import BIRD_EXAMPLES, Spider_EXAMPLES


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--result_directory', type=str, required=True, help="Directory stores the candidates")
    parser.add_argument('--config', type=str, default="./run/configs/selector_config.yaml",
                        help="Path to the configuration file.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    parser.add_argument('--pick_final_sql', type=bool, default=False,
                        help="Pick the final SQL from the generated SQLs.")

    parser.add_argument('--base_url', type=str, default="http://localhust:8000", help="Base url for openAI API call")
    parser.add_argument('--engine_name', type=str, default='qwen2.5-coder-7b', help="Engine name for LLM call")

    args = parser.parse_args()
    args.run_start_time = datetime.now().isoformat()
    with open(args.config, 'r') as file:
        args.config = file.read()
        args.config = yaml.safe_load(
            Template(args.config).render({"default_engine": args.engine_name, "base_url": args.base_url}))
    return args


def create_state(question_id, db_id, data_item, args):
    state_values = SystemState(
        task=Task(question_id=question_id, db_id=db_id, question=data_item[-1]['final_SQL']['Question'],
                  evidence=data_item[-1]['final_SQL']['Evidence'],
                  SQL=data_item[-1]['final_SQL']['GOLD_SQL'],
                  examples=Spider_EXAMPLES if "spider" in args.data_path else BIRD_EXAMPLES,
                  difficulty=None),
        tentative_schema=DatabaseManager(db_mode=args.data_mode, db_id=db_id).get_db_schema(),
        keywords=data_item[0]['keywords'],
        similar_columns=data_item[1]['similar_columns'],
        schema_with_examples=data_item[1]['schema_with_examples'],
        schema_with_descriptions=data_item[2]['schema_with_descriptions'],
        execution_history=[])
    return state_values


def file_worker(file_name: str, config, result_directory):
    db_id = re.findall("_(.*?)\.json$", file_name)[0]
    question_id = int(re.findall(r"^(\d+)_.*", file_name)[0])
    logger = Logger(db_id=db_id, question_id=str(question_id),
                    result_directory=str(result_directory) + "/selection_logs/")
    file_path = os.path.join(result_directory, file_name)

    with open(file_path, 'r', encoding='utf8') as f:
        data_item = json.load(f)
        try:
            candidates = next((item.get("candidates", []) for item in data_item if item.get("tool_name") == "revise"),
                              [])
            state = create_state(question_id, db_id, data_item, args)
            if len(candidates) == 0:
                print("Question {} has no candidates been generated!".format(question_id))
                return {
                    "db_id": db_id,
                    "question_id": question_id,
                    "ex": (False, False, False),
                    "selected_sql": None,
                    "gold_sql": state.task.SQL if 'state' in locals() else None,
                    "correct_portion": 0.0
                }
            else:
                candidates = [candidate.get("refined_query", "") for candidate in candidates]
                sql_metaInfos = [SQLMetaInfo(SQL=item) for item in candidates]
        except Exception as e:
            print("error : {}".format(e))
            return {
                "db_id": db_id,
                "question_id": question_id,
                "ex": (False, False, False),
                "selected_sql": None,
                "gold_sql": state.task.SQL if 'state' in locals() else None,
                "correct_portion": 0.0
            }

        all_candidate_results = []
        for candidate_sql in candidates:
            response = DatabaseManager().compare_sqls(
                predicted_sql=candidate_sql,
                ground_truth_sql=state.task.SQL,
            )
            if response['exec_res'] == 1:
                all_candidate_results.append(True)
            else:
                all_candidate_results.append(False)
        upper_bound, lower_bound = any(all_candidate_results), all(all_candidate_results)
        if not upper_bound:
            prediction = False
        elif lower_bound:
            prediction = True
        else:
            try:
                selection_method = SelectionMethodFactory.get_selection_method()
                selected_sql = selection_method(state, sql_metaInfos, config, args)
                response = DatabaseManager().compare_sqls(
                    predicted_sql=selected_sql,
                    ground_truth_sql=state.task.SQL,
                )
                if response['exec_res'] == 1:
                    prediction = True
                else:
                    prediction = False
            except Exception as e:
                print("Error in API Call with question id {}, error message: {}".format(question_id, e))
                prediction = lower_bound
        result_dict = {
            "db_id": db_id,
            "question_id": question_id,
            "ex": (prediction, upper_bound, lower_bound),
            "selected_sql": locals().get("selected_sql", None),
            "gold_sql": state.task.SQL,
            "correct_portion": np.sum(all_candidate_results) / len(all_candidate_results)
        }
        return result_dict


def update_progress(pbar, all_predict: list, prediction_result: list, result_directory, args):
    def callback(result):
        pbar.update(1)

        prediction, upper_bound, lower_bound = result["ex"]

        if lower_bound:
            status = "TRUE_OF_ALL_CANDIDATES"
        elif not upper_bound:
            status = "FALSE_OF_ALL_CANDIDATES"
        else:
            if not prediction:
                status = "FALSE_OF_SELECTION"
            else:
                status = "TRUE_OF_SELECTION"

        all_predict.append((prediction, upper_bound, lower_bound))

        print_result(all_predict)

        record = {
            "db_id": result["db_id"],
            "question_id": result["question_id"],
            "status": status,
            "selected_sql": result["selected_sql"],
            "gold_sql": result["gold_sql"],
            "correct_portion": result['correct_portion']
        }
        prediction_result.append(record)
        if not os.path.exists(os.path.join(str(result_directory), 'selection_logs')):
            os.makedirs(os.path.join(str(result_directory), 'selection_logs'))
        with open(str(result_directory) + '/selection_logs/-{}_selection_result.json'.format(args.selection_method),
                  'w', encoding='utf8') as f:
            json.dump(prediction_result, f, indent=4, ensure_ascii=False)

    return callback


def print_result(all_predict):
    print("\n\n---------------------------------------Execution Accuracy-----------------------------------")

    header = "{:<12} {:>10} {:>22} {:>22} {:>22}".format("Metric", "Value", "Prediction", "Upper Bound", "Lower Bound")
    separator = "-" * len(header)
    print(header)
    print(separator)

    total_count = len(all_predict) if all_predict else 0
    prediction_count = sum(1 for pred, _, _ in all_predict if pred == 1) if all_predict else 0
    upper_count = sum(1 for _, upper, _ in all_predict if upper == 1) if all_predict else 0
    lower_count = sum(1 for _, _, lower in all_predict if lower == 1) if all_predict else 0

    prediction_accuracy = (prediction_count / total_count) * 100 if total_count > 0 else 0.0
    upper_accuracy = (upper_count / total_count) * 100 if total_count > 0 else 0.0
    lower_accuracy = (lower_count / total_count) * 100 if total_count > 0 else 0.0

    row_format = "{:<12} {:>10} {:>15.2f}% ({:>3}) {:>15.2f}% ({:>3}) {:>15.2f}% ({:>3})"
    print(row_format.format(
        "Accuracy",
        total_count,
        prediction_accuracy, prediction_count,
        upper_accuracy, upper_count,
        lower_accuracy, lower_count
    ))
    print(separator + "\n\n")


if __name__ == '__main__':
    args = parse_arguments()
    result_directory = Path("./results") / args.data_mode / args.config['setting_name'] / Path(
        args.data_path).stem / args.result_directory
    if not os.path.exists(result_directory):
        raise "Candidate result  directory \"{}\" do not exist!".format(result_directory)
    config = args.config
    config['result_dir'] = result_directory
    excluded_patterns = ["-args.json", "-predictions.json", "-statistics.json", "selection_data.json"]
    all_predict = []
    prediction_result = []
    POOL_NUM = args.num_workers
    ##########Run###############
    file_names = sorted([f for f in os.listdir(result_directory) if
                         f.endswith(".json") and not any(pattern in f for pattern in excluded_patterns)])
    ############################
    random.shuffle(file_names)
    with tqdm(total=len(file_names), desc="Generating:") as pbar:
        if POOL_NUM > 1:
            with Pool(POOL_NUM) as pool:
                for file_name in file_names:
                    pool.apply_async(file_worker, args=(file_name, config, result_directory),
                                     callback=update_progress(pbar, all_predict, prediction_result, result_directory,
                                                              args))
                pool.close()
                pool.join()
        else:
            for file_name in file_names:
                result = file_worker(file_name, config, result_directory)
                update_progress(pbar, all_predict, prediction_result, result_directory, args)(result)
    print("\n\n--------------------------------------***Final Result***----------------------------------")
    print_result(all_predict)
