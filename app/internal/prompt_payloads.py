from typing import List, Dict, Any
import json



##survey question payloads to construct:  input for all functions is the dict extracted from survey, output is a string
# short/long answer question prompts
# multiple choice question prompts
# checkbox question prompts 
# linear scale question prompts

answer_schema = {"answer": ""}

def short_answer_prompt(question_dict: Dict[str, Any], JSON: bool) -> str:
    question_text: str = question_dict["question"]
    answer_format: str = question_dict["answer_format"]
    question_payload = question_text + f"\n Your response must be a short answer of type {answer_format}."
    if JSON:
        json_payload = question_payload + "\n JSON response schema:\n" + json.dumps(answer_schema)
        return json_payload
    return question_payload

def long_answer_prompt(question_dict: Dict[str, Any], JSON: bool) -> str:
    question_text: str = question_dict["question"]
    answer_format: str = question_dict["answer_format"]
    question_payload = question_text + f"\n Your response must be a long answer of type {answer_format}."
    if JSON:
        json_payload = question_payload + "\n JSON response schema:\n" + json.dumps(answer_schema)
        return json_payload
    return question_payload

def ranking_prompt(question_dict: Dict[str, Any], JSON: bool) -> str:
    question_text: str = question_dict["question"]
    choices: List[str] = question_dict["choices"]
    question_payload = question_text + "The choices are:\n" + "\n".join([f"{option}" for option in choices]) + "\n Your response must be a ranked list of the choices given."
    if JSON:
        json_payload = question_payload + "\n JSON response schema:\n" + json.dumps(answer_schema)
        return json_payload
    return question_payload

def multiple_choice_prompt(question_dict: Dict[str, Any], JSON: bool) -> str:
    question_text: str = question_dict["question"]
    choices: List[str] = question_dict["choices"]
    question_payload = question_text + "The choices are:\n" + "\n".join([f"{option}" for option in choices]) + "\n Your response must only be one of the choices from the list of choices given."
    if JSON:
        json_payload = question_payload + "\n JSON response schema:\n" + json.dumps(answer_schema)
        return json_payload
    return question_payload


def checkbox_prompt(question_dict: Dict[str, Any], JSON: bool) -> str:
    question_text: str = question_dict["question"]
    choices: List[str] = question_dict["choices"]
    question_payload = question_text + "The choices are:\n" + "\n".join([f"{option}" for option in choices]) + "\n Your response must be one or multiple choices from the list of choices given."
    if JSON:        
        json_payload = question_payload + "\n JSON response schema:\n" + json.dumps(answer_schema)
        return json_payload
    return question_payload

def linear_scale_prompt(question_dict: Dict[str, Any], JSON: bool) -> str:
    question_text: str = question_dict["question"]
    min_value: int = question_dict["min_value"]
    max_value: int = question_dict["max_value"]
    if min_value < 0:
        min_value = 0
    if max_value > 10:
        max_value = 10
    question_payload = question_text + "\n" + f"Scale: {list(range(min_value, max_value+1))}" + "\n Your response must be a an integer on the scale given"
    if JSON:
        json_payload = question_payload + "\n JSON response schema:\n" + json.dumps(answer_schema)
        return json_payload
    return question_payload


## agent initialization prompt: input is demographic parameters, initial world state/ instructions from survey body, output is a string
def initialization_prompt(demographic: Dict, persona: str) -> str: 
    demographic_information = '\n'.join([f"{k}: {v}" for k, v in demographic.items()])
    prompt_payload = f"You are not a helpful assistant. You are a person with the following identity:\n"+f"{demographic_information}\n\n" + f"You should behave according to the persona given: \n {persona}" + "You must think and behave as this identity when responding to all queries."
    return prompt_payload

    
class Iterator:

    def __init__(self, json_mode: bool, iteration_questions: list[dict]):

        self.json_mode = json_mode
        self.iterations = iteration_questions
        self.n_of_iter = len(iteration_questions)
        self.current_index = 0 
    

    def input_question(self,question: Dict) -> str:

        if question["type"] == "short answer":
            return short_answer_prompt(question, self.json_mode)
        
        elif question["type"] == "long answer":
            return long_answer_prompt(question,self.json_mode)

        elif question["type"] == "multiple choice":
            return multiple_choice_prompt(question, self.json_mode)
            
        elif question["type"] == "checkboxes":
            return checkbox_prompt(question, self.json_mode)
        
        elif question["type"] == "ranking":
            return ranking_prompt(question, self.json_mode)
        
        elif question["type"] == "linear scale":
            return linear_scale_prompt(question, self.json_mode)
        
        else:
            return str(question["question"])


    def iter(self) -> str:
        if self.current_index < self.n_of_iter:
            question = self.iterations[self.current_index]
            self.current_index += 1  # Move to the next question
            return self.input_question(question)
        else:
            return StopIteration
