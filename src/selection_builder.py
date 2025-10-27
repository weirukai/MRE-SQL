import itertools
import json
import os
import random
import re
from langchain_core.exceptions import OutputParserException

from workflow.agents.unit_tester.tool_kit.generate_unit_test import GenerateUnitTest
from workflow.agents.unit_tester.tool_kit.evaluate import Evaluate
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from typing import List
from database_utils.execution import ExecutionStatus
from collections import defaultdict, Counter
from itertools import combinations, product
from runner.database_manager import DatabaseManager
from llm.models import async_llm_chain_call, get_llm_chain, call_llm_chain_multi_sampling
from llm.parsers import get_parser
from llm.prompts import get_prompt
import traceback
import math
import numpy as np
from ensemble_builder import uncertainty_based_ensemble_v1, uncertainty_based_ensemble_new_v2, \
    uncertainty_based_ensemble_v3, uncertainty_based_ensemble_v4, uncertainty_based_ensemble_v5, \
    uncertainty_based_ensemble
from datetime import datetime

import time
from functools import wraps




def binary_ranking_selection(state: SystemState, sql_metaInfos: List[SQLMetaInfo], config, args) -> str:
    """
        select the best candidate in the listwise manner
        """
    indexed_infos = {i: info for i, info in enumerate(sql_metaInfos)}
    correct_counts = defaultdict(int)
    pairs = list(combinations(range(len(sql_metaInfos)), 2))
    request_list = []
    for idx1, idx2 in pairs:
        c1, c2 = indexed_infos[idx1], indexed_infos[idx2]
        try:
            if c1.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT and c2.execution_status != ExecutionStatus.SYNTACTICALLY_CORRECT:
                correct_counts[idx1] += 1
            elif c1.execution_status != ExecutionStatus.SYNTACTICALLY_CORRECT and c2.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT:
                correct_counts[idx2] += 1
            elif (c1.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT and
                  c2.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT and
                  DatabaseManager().compare_sqls(c1.SQL, c2.SQL)['exec_res'] == 1):
                correct_counts[idx1] += 1
            elif (c1.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT and
                  c2.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT and
                  DatabaseManager().compare_sqls(c1.SQL, c2.SQL)['exec_res'] == 0):
                request_kwargs = {
                    "DATABASE_SCHEMA": state.get_schema_string(schema_type="tentative"),
                    "QUESTION": state.task.question,
                    "HINT": state.task.evidence,
                    "CANDIDATE_A_QUERY": c1.SQL,
                    "CANDIDATE_A_RESULT": DatabaseManager().execute_sql(c1.SQL, fetch=10),
                    "CANDIDATE_B_QUERY": c2.SQL,
                    "CANDIDATE_B_RESULT": DatabaseManager().execute_sql(c1.SQL, fetch=10),
                    "Examples": state.task.examples,
                }
                request_list.append((idx1, idx2, request_kwargs))
        except Exception as e:
            print(f"Prepare request list error: {e}")

    if request_list:
        generator_config = config['team_agents']['binary_selector']['tools']['binary_selection']
        responses = async_llm_chain_call(
            prompt=get_prompt(template_name=generator_config['template_name']),
            engine=get_llm_chain(**generator_config['engine_config']),
            parser=get_parser(generator_config['parser_name']),
            request_list=[rk for (_, _, rk) in request_list],
            step=f"selection_{generator_config['engine_config']['engine_name']}",
        )
        responses = [item for item in responses if "OutputParserException" not in str(item)]
        if not responses:
            raise OutputParserException("Invalid response format from binary selection")
        for (idx1, idx2, _), response in zip(request_list, responses):
            response = str(response[0]).upper()
            if response == "A":
                correct_counts[idx1] += 1
            elif response == "B":
                correct_counts[idx2] += 1
    if not correct_counts:
        raise ValueError("No valid candidates found after comparison")
    best_idx = max(correct_counts.items(), key=lambda x: x[1])[0]
    best_candidate = indexed_infos[best_idx].SQL
    state.SQL_meta_infos['BS_Ranks'] = {
        f"{info.SQL}": count
        for idx, count in correct_counts.items()
        for info in [indexed_infos[idx]]
    }
    return best_candidate


def listwise_ranking_selection(state: SystemState, sql_metaInfos: List[SQLMetaInfo], config, args) -> str:
    """
    select the best candidate in the listwise manner
    """
    sql_metaInfos = [sql_metaInfo for sql_metaInfo in sql_metaInfos if
                     sql_metaInfo.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT]

    ###########################################################
    formatted_candidates = ""
    for i, item in enumerate(sql_metaInfos):
        formatted_candidates += f"SQL {i + 1}: {item.SQL} \n Execution result: {DatabaseManager().execute_sql(item.SQL, fetch=10)}"
        formatted_candidates += "\n\n===============\n\n"
    ###############
    request_kwargs = {
        "DATABASE_SCHEMA": state.get_schema_string(schema_type="tentative"),
        "QUESTION": state.task.question,
        "HINT": state.task.evidence,
        "Examples": state.task.examples,
        "Candidates": formatted_candidates
    }
    generator_config = config['team_agents']['listwise_selector']['tools']['listwise_selection']
    responses = call_llm_chain_multi_sampling(
        prompt=get_prompt(
            template_name=generator_config['template_name']),
        engine=get_llm_chain(**generator_config['engine_config'], sampling_count=generator_config['sampling_count']),
        parser=get_parser(generator_config['parser_name']),
        request_kwargs=request_kwargs,
        step=f"selection_{generator_config['engine_config']['engine_name']}",
    )
    # filter out parser error####
    responses = [item for item in responses if item != "OutputParserException"]
    #############################
    if len(responses) == 0:
        raise OutputParserException(
            "Your listwise selection answer is not in the correct format. Please make sure to include your answer in the JSON format")
    else:
        state.SQL_meta_infos['LS_Selection'] = responses
        final_response = SC_result_selection(state, [SQLMetaInfo(SQL=item) for item in responses], config, args)
    return final_response


def groupwise_ranking_selection(state: SystemState, sql_metaInfos: List[SQLMetaInfo], config, args) -> str:
    """
    select the best candidate in the groupwise manner
    """
    sql_metaInfos = [sql_metaInfo for sql_metaInfo in sql_metaInfos if
                     sql_metaInfo.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT]

    ##################################################
    clusters = execution_based_clustering(sql_metaInfos)
    ##################################################

    sorted_cluster_items = sorted(
        clusters.items(),
        key=lambda item: len(item[1]),
        reverse=True
    )

    sorted_lists = [v for k, v in sorted_cluster_items]
    combinations = [list(combo) for combo in product(*sorted_lists)]

    #### fast debug#######
    random.shuffle(combinations)
    combinations = combinations[:config['team_agents']['groupwise_selector']['tools'][
        'groupwise_selection']['sampling_count']]
    ###########
    request_list = []
    for combination in combinations:
        formatted_candidates = ""
        for i, item in enumerate(combination):
            formatted_candidates += f"SQL {i + 1}: {item.SQL} \n Execution result: {DatabaseManager().execute_sql(item.SQL, fetch=10)}"
            formatted_candidates += "\n\n===============\n\n"
        request_kwargs = {
            "DATABASE_SCHEMA": state.get_schema_string(schema_type="tentative"),
            "QUESTION": state.task.question,
            "HINT": state.task.evidence,
            "Examples": state.task.examples,
            "Candidates": formatted_candidates
        }
        request_list.append(request_kwargs)
    generator_config = config['team_agents']['groupwise_selector']['tools']['groupwise_selection']
    responses = async_llm_chain_call(
        prompt=get_prompt(
            template_name=generator_config['template_name']),
        engine=get_llm_chain(**generator_config['engine_config']),
        parser=get_parser(generator_config['parser_name']),
        request_list=request_list,
        step=f"selection_{generator_config['engine_config']['engine_name']}",
    )
    # filter out parser error####
    responses = [item[0] for item in responses if item[0] != "OutputParserException" and len(item[0]) > 10]
    #############################
    if len(responses) == 0:
        raise OutputParserException(
            "Your groupwise selection answer  is not in the correct format. Please make sure to include your answer in the JSON format")
    else:
        state.SQL_meta_infos['GS_Selection'] = responses
        final_response = SC_result_selection(state, [SQLMetaInfo(SQL=item) for item in responses], config, args)
    return final_response


def point_wise_selection(state: SystemState, sql_metaInfos: List[SQLMetaInfo], config, args) -> str:
    request_list = []
    SYNTACTICALLY_CORRECT_Candidates = []
    for sql_metaInfo in sql_metaInfos:
        if sql_metaInfo.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT:
            request_kwargs = {
                "DATABASE_SCHEMA": state.get_schema_string(schema_type="tentative"),
                "QUESTION": state.task.question,
                "HINT": state.task.evidence,
                "Candidate": {"SQL": sql_metaInfo.SQL,
                              "Execution result": DatabaseManager().execute_sql(sql_metaInfo.SQL, fetch=10)}
            }
            request_list.append(request_kwargs)
            SYNTACTICALLY_CORRECT_Candidates.append(sql_metaInfo)
    if len(request_list) > 0:
        generator_config = config['team_agents']['pointwise_selector']['tools']['pointwise_selection']

        responses = async_llm_chain_call(
            prompt=get_prompt(
                template_name=generator_config['template_name']),
            engine=get_llm_chain(**generator_config['engine_config']),
            parser=get_parser(generator_config['parser_name']),
            request_list=request_list,
            step=f"selection_{generator_config['engine_config']['engine_name']}",
        )

        ###parser error##
        responses = [item for item in responses if "OutputParserException" not in item]
        ################
        if len(responses) == 0:
            raise OutputParserException(
                "Your pointwise seleciton answer  is not in the correct format. Please make sure to include your answer in the format <Answer>...</Answer>")
        else:
            candidate_pointwise = []
            for i, response in enumerate(responses):
                if "True" in response:
                    candidate_pointwise.append(SYNTACTICALLY_CORRECT_Candidates[i])
            state.SQL_meta_infos["PS_Selections"] = candidate_pointwise
            final_sql = SC_result_selection(state, candidate_pointwise, config, args)
            return final_sql


def multi_agent_selection(state: SystemState, sql_metaInfos: List[SQLMetaInfo], config, args) -> str:
    candidates = []
    candidates_without_errors = [sql_metaInfo.SQL for sql_metaInfo in sql_metaInfos if
                                 sql_metaInfo.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT]

    #################################################
    clusters = execution_based_clustering(sql_metaInfos)
    BS_result_to_max_count, PS_Clusters, GS_Clusters, LS_Clusters, UT_result_to_max_count = None, None, None, None, None
    ################################pairwise selection ###################################
    try:
        candidate_1 = binary_ranking_selection(state, sql_metaInfos, config, args)
        candidates.append(candidate_1)
        BS_result_to_max_count = defaultdict(int)
        for sql, count in state.SQL_meta_infos['BS_Ranks'].items():
            try:
                execute_result = str(SQLMetaInfo(SQL=sql).execution_result)
            except Exception as e:
                continue
            if count > BS_result_to_max_count[execute_result]:
                BS_result_to_max_count[execute_result] = count
    except Exception:
        candidate_1 = random.choice(candidates_without_errors)
        candidates.append(candidate_1)
        print("Question {}, BS failed".format(state.task.question_id))
        pass
    ################################pointwise selection ###################################
    try:
        candidate_2 = point_wise_selection(state, sql_metaInfos, config, args)
        candidates.append(candidate_2)
        PS_Clusters = execution_based_clustering(state.SQL_meta_infos['PS_Selections'])
    except Exception:
        try:
            candidate_2 = point_wise_selection(state, sql_metaInfos, config, args)
            candidates.append(candidate_2)
            PS_Clusters = execution_based_clustering(state.SQL_meta_infos['PS_Selections'])
        except Exception:
            candidate_2 = random.choice(candidates_without_errors)
            candidates.append(candidate_2)
            print("Question {}, PS failed".format(state.task.question_id))
    ################################groupwie selection###################################
    try:
        candidate_3 = groupwise_ranking_selection(state, sql_metaInfos, config, args)
        candidates.append(candidate_3)
        GS_Selections = state.SQL_meta_infos['GS_Selection']
        GS_Clusters = execution_based_clustering([SQLMetaInfo(SQL=item) for item in GS_Selections])
    except Exception:
        candidate_3 = random.choice(candidates_without_errors)
        candidates.append(candidate_3)
        print("Question {}, GS failed".format(state.task.question_id))
    ################################listwise selection ###################################
    try:
        candidate_4 = listwise_ranking_selection(state, sql_metaInfos, config, args)
        candidates.append(candidate_4)
        LS_selections = state.SQL_meta_infos['LS_Selection']
        LS_Clusters = execution_based_clustering([SQLMetaInfo(SQL=item) for item in LS_selections])
    except Exception:
        candidate_4 = random.choice(candidates_without_errors)
        candidates.append(candidate_4)
        print("Question {}, LS failed".format(state.task.question_id))
    ################################UT selection ###################################
    try:
        candidate_5 = Unit_test_selection(state, sql_metaInfos, config, args)
        candidates.append(candidate_5)
        UT_selections = state.SQL_meta_infos['UT_Selection']
        UT_result_to_max_count = defaultdict(int)
        for sql, count in UT_selections.items():
            try:
                execute_result = str(SQLMetaInfo(SQL=sql).execution_result)
            except Exception as e:
                continue
            if count > UT_result_to_max_count[execute_result]:
                UT_result_to_max_count[execute_result] = count
    except Exception:
        ## try again
        try:
            candidate_5 = Unit_test_selection(state, sql_metaInfos, config, args)
            candidates.append(candidate_5)
            UT_selections = state.SQL_meta_infos['UT_Selection']
            UT_result_to_max_count = defaultdict(int)
            for sql, count in UT_selections.items():
                try:
                    execute_result = str(SQLMetaInfo(SQL=sql).execution_result)
                except Exception as e:
                    continue
                if count > UT_result_to_max_count[execute_result]:
                    UT_result_to_max_count[execute_result] = count
        except Exception:
            candidate_5 = random.choice(candidates_without_errors)
            candidates.append(candidate_5)
            print("Question {}, UT failed. {}".format(state.task.question_id, traceback.format_exc()))


    if len(candidates) > 0:
        args.NUM_CANDIDATE, args.NUM_TRY, args.NUM_TEST = len(sql_metaInfos), \
                                                          config['team_agents']['listwise_selector']['tools'][
                                                              'listwise_selection']['sampling_count'], \
                                                          config['team_agents']['unit_tester']['tools'][
                                                              'generate_unit_test']['unit_test_count'] + 1

        final_result = uncertainty_based_ensemble(state, clusters, BS_Selections=BS_result_to_max_count,
                                                  PS_Selections=PS_Clusters, GS_Selections=GS_Clusters,
                                                  LS_Selections=LS_Clusters, UT_Selections=UT_result_to_max_count,
                                                  args=args)
    else:
        final_result = SC_result_selection(state, sql_metaInfos, config, args)
    return final_result




def SC_result_selection(state, sql_metaInfos: List[SQLMetaInfo], config, args) -> str:
    """
       self-consistency for candidate selection
    """

    sql_metaInfos = [sql_metaInfo for sql_metaInfo in sql_metaInfos if
                     sql_metaInfo.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT]

    clusters = execution_based_clustering(sql_metaInfos)
    if not clusters:
        raise ValueError("No clusters found, check input sql_metaInfos.")
    max_cluster = max(clusters.values(), key=lambda group: len(group))
    selected_sql_meta = max_cluster[0]
    return selected_sql_meta.SQL


def Unit_test_selection(state: SystemState, sql_metaInfos: List[SQLMetaInfo], config, args):
    unit_tester = GenerateUnitTest(**config['team_agents']['unit_tester']["tools"]["generate_unit_test"])

    evaluator = Evaluate(**config['team_agents']['unit_tester']["tools"]["evaluate"])
    state.SQL_meta_infos['candidates'] = [sql_metaInfo for sql_metaInfo in sql_metaInfos if
                                          sql_metaInfo.execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT]
    try:
        unit_tester._run(state)
    except:
        unit_tester._run(state)
    evaluator._run(state)
    state.SQL_meta_infos['UT_Selection'] = dict(
        zip([item.SQL for item in state.SQL_meta_infos['candidates']], evaluator.scores))
    return state.SQL_meta_infos['evaluate_1'][0].SQL




def execution_based_clustering(candidate_queries: List[SQLMetaInfo]) -> dict:
    """
    Clusters the generated candidates based on the execution results.
    Args:
        state (SystemState): The current system state.
    """
    clusters = {}
    exceptions = []
    for query in candidate_queries:
        try:
            result = str(query.execution_result) if isinstance(query.execution_result, str) else repr(
                query.execution_result)
        except Exception as e:
            exceptions.append(str(e))
            continue
        if result not in clusters:
            clusters[result] = []
        clusters[result].append(query)
    # sample one query from each cluster
    if not clusters:
        clusters["\n".join(exceptions)] = candidate_queries
    return clusters



class SelectionMethodFactory:
    @staticmethod
    def get_selection_method():
        return multi_agent_selection

