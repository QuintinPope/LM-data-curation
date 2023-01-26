import pandas as pd
import openai
import requests
import os
import json

# Load that data into memory as a list of dictionaries:
def load_jsonl_file(path):
    f = open(path)
    json_lines_list = []
    for line in f.readlines():
        json_lines_list.append(json.loads(line))
    f.close()
    return json_lines_list


def load_zero_shot_instructions(path):
    instructions = open(path, "r").read()
    return instructions


# Constructs the full prompt from the combination of the initial curation prompt, the few shot examples, and the query to be decided:
def compose_full_prompt(zero_shot_instructions, few_shot_examples = [], query = "", input_name="input", first_response_name="analysis"):
    few_shot_prompt = zero_shot_instructions + "\n"
    for shot, i in zip(few_shot_examples, range(len(few_shot_examples))):
            few_shot_prompt = few_shot_prompt + "\n"
            for key in shot.keys():
                few_shot_prompt = few_shot_prompt + key + ": " + shot[key] + "\n"
    full_prompt = few_shot_prompt + "\n" + input_name + ": " + query + "\n" + first_response_name + ":"
    return full_prompt


def curation_decision(full_prompt, 
                      model, 
                      target_length, 
                      temperature, 
                      n, 
                      output_prompt_text="decision",
                      positive_decision_str="include",
                      negative_decision_str="exclude"):
    output_prompt_text = str.lower(output_prompt_text)
    positive_decision_str = str.lower(positive_decision_str)
    negative_decision_str = str.lower(negative_decision_str)

    completion = openai.Completion.create(
        engine=model,
        prompt=full_prompt,
        max_tokens=target_length,
        temperature=temperature,
        n=n,
        logprobs=5,
    )
    first_completion_text = completion['choices'][0]['text']
    decision_text = first_completion_text
    full_interaction = full_prompt + first_completion_text

    positive_decision_indicator = output_prompt_text + ": " + positive_decision_str
    negative_decision_indicator = output_prompt_text + ": " + negative_decision_str

    if not (positive_decision_indicator in str.lower(full_interaction) or negative_decision_indicator in str.lower(full_interaction)):
        print("Error: \"" + str.lower(full_interaction) + "\" did not contain either:")
        print(positive_decision_indicator)
        print("or:")
        print(negative_decision_indicator)
        second_completion = openai.Completion.create(
        engine=model,
        prompt=full_prompt + first_completion_text + "\n" + output_prompt_text + ":",
        max_tokens=2,
        temperature=temperature,
        n=n,
        logprobs=5,
        )
        second_completion_text = second_completion['choices'][0]['text']
        decision_text = first_completion_text + "\n" + output_prompt_text + ":" + second_completion_text
        full_interaction = full_prompt + decision_text


    if positive_decision_indicator in str.lower(full_interaction):
        decision = positive_decision_str
    elif negative_decision_indicator in str.lower(full_interaction):
        decision = negative_decision_str
    else:
        decision = "no decision"

    return decision, decision_text, full_interaction

def run_curation_of_anthropic_data(anthropic_data, 
                                   zero_shot_prompt, 
                                   few_shot_examples=[], 
                                   model="text-curie-001",
                                   target_length=50,
                                   temperature=0,
                                   n=1,
                                   output_prompt_text="decision",
                                   positive_decision_str="include",
                                   negative_decision_str="exclude",
                                   input_name="input", 
                                   first_response_name="analysis"):
    anthropic_data_decisions = []
    for line in anthropic_data:
        statement = line['statement']
        full_prompt = compose_full_prompt(zero_shot_instructions=zero_shot_prompt, 
                                          few_shot_examples=few_shot_examples, 
                                          query=statement,
                                          input_name=input_name,
                                          first_response_name=first_response_name)
        
        decision, decision_text, full_interaction = curation_decision(full_prompt, 
                                                                      model, 
                                                                      target_length, 
                                                                      temperature, 
                                                                      n,
                                                                      output_prompt_text=output_prompt_text,
                                                                      positive_decision_str=positive_decision_str,
                                                                      negative_decision_str=negative_decision_str)
        
        anthropic_data_decisions.append([statement, decision, decision_text, full_interaction])
    return anthropic_data_decisions

def compare_two_curation_decision_sets(set_1, set_1_name, set_2, set_2_name, only_show_disagreements=False):
    for set_1_result, set_2_result in zip(set_1, set_2):
        if set_1_result[1] == set_2_result[1]:
            if only_show_disagreements:
                print("Agreed on input: " + set_1_result[0] + "\n\n")
                continue
        print("Input: " + set_1_result[0] + "\n")
        print(set_1_name + " result:")
        print("Decision: " + set_1_result[1])
        print("Model decision text:\n" + set_1_result[2] + "\n")

        print(set_2_name + " results:")
        print("Decision: " + set_2_result[1])
        print("Model decision text:\n" + set_2_result[2])
        print("\n\n")