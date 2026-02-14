"""LLM for summarising observations in the history for the AI2Thor environment"""
import openai
from openai.error import RateLimitError, APIConnectionError
from time import sleep

summariser_prompt = """You are an excellent summarizer of text. You are helping another human being in carrying out a task. The human being will give you a list of objects that it is observing and the task it is trying to carry out. You need to summarize out relevant information from the observations that is necessary for the human to carry out the task efficiently. The objects in the observation are of the form "<object_name>_<object_id>".

Input format: {Task Description: description of the task the human is trying to achieve, Observation: list of objects that the human is observing in the current time step}
Output format: List of relevant observations that could be explored or interacted with in the environment in the current time step or in the future."""

import json
import openai
from openai.error import RateLimitError, APIConnectionError
from time import sleep

import os, sys

sys.path.append(os.path.abspath((os.getcwd())))

from AI2Thor.strict_output import strict_output


class SummaryLLM(object):
    def __init__(self, summary_llm_cache, model, temperature=0) -> None:
        self.summary_llm_cache = summary_llm_cache
        self.model = model
        self.temperature = temperature

    def gpt_call(self, **kwargs):
        """MODEL = "gpt-3.5-turbo-0613"
        MODEL = "gpt-4-0314"
        """
        cache_id = self.get_hashable_id(**kwargs)
        if cache_id in self.summary_llm_cache.keys():
            print("LLM cache hit, returning")
            return self.summary_llm_cache[cache_id]
        else:
            while True:
                try:
                    response = openai.ChatCompletion.create(**kwargs)
                    break
                except (RateLimitError, APIConnectionError) as e:
                    print(e)
                    sleep(10)
            self.action_llm_cache[cache_id] = response
            return response, cache_id

    def get_gpt_response(self, user_prompt, output_format):
        response = strict_output(
            summariser_prompt,
            user_prompt,
            output_format,
            model=self.model,
            temperature=self.temperature,
        )
        action = list(response.values())
        return action

    def get_hashable_id(self, **kwargs):
        """Get a hashable id for the kwargs"""
        hashable = ()
        for value in kwargs.values():
            if isinstance(value, list):
                value = json.dumps(value)
            hashable += (value,)
        return hashable


# TODO update with object_id and receptacle_id definition in prompt.
if __name__ == "__main__":
    llm_cache = {}
    model = "gpt-4-0314"
    llm = SummaryLLM(llm_cache, model)
    user_prompt = {
        "Task Description": "bring a tomato, lettuce and bread to the countertop to make a sandwich",
        "Observation": [
            "Cabinet_1",
            "UpperCabinets_1",
            "StandardCounterHeightWidth_1",
            "Wall_1",
            "Ceiling_1",
            "Fridge_1",
            "Toaster_1",
            "CoffeeMachine_1",
            "CounterTop_1",
        ],
    }
    # should return ["Cabinet_1", "UpperCabinets_1", "Fridge_1", "CounterTop_1"]
    output_format = {"Observation": ["refrigerator", "cabinet", "glass of water"]}
    response = llm.get_gpt_response(user_prompt, output_format)
    print(response)
