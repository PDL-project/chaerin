"""LLM for summarising observations in the history for the AI2Thor environment"""

import openai
from time import sleep
import json
import os, sys
import requests
import ast
import re

summariser_prompt = """You are an excellent summarizer of text. You are helping another human being in carrying out a task. 
The human being will give you a list of objects that they is observing and the task they are trying to carry out. 
You need to summarize out relevant information from the observations that is necessary for the human to carry out the task efficiently. 
The objects in the observation are of the form "<object_name>_<object_id>".

Input format: {{Task: description of the task the human is trying to achieve, Observation: list of objects that the human is observing at the current time step (can also be an empty list)}}
Output format: List of relevant observations that could be explored or interacted with in the environment in the current time step or in the future.

Here is an example of what the FORMAT an of an output looks like:
"DeskLamp_6", "cheese_2", "Table_4", "LightSwitch_5"

--> If the list of objects that the human is observing at the current time is empty just return empty

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
"""


sys.path.append(os.path.abspath((os.getcwd())))


class ObsSummaryLLM(object):
    def __init__(self, objective_input, headers, temperature=0) -> None:
        """
        objective_input (str): the task to be carried out by the agents
        headers (dict): Openai credentials
        """
        self.objective_input = self._formater(objective_input)
        self.temperature = temperature

        # with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
        #     key = json.load(json_file)
        #     api_key = key["my_openai_api_key"]
        # headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.headers = headers

    def _formater(self, input_prompt):
        """
        parameter:
            input_prompt (str): the task to be carried out by the agents

        return:
            filtered_input (str): task string with "Task: " removed
        """
        pattern = r"^Task: "
        filtered_input = re.sub(pattern, " ", input_prompt, flags=re.I)
        return filtered_input

    def prepare_prompt(self, observations):
        """
        parameters:
            observations (str): observations from agent's point of view

        return:
            prompt (str): combined task and observation string to fed into LLM
        """
        base_dict = {"Task": self.objective_input}
        obs_dict = {"Observation": observations}
        combined = {**base_dict, **obs_dict}

        prompt = json.dumps(combined)
        return prompt

    def prepare_payload(self, observations):
        """
        parameters:
            observations (str): observations from agent's point of view

        return:
            payload (json): gpt payload, including system and user prompts
        """
        prompt = self.prepare_prompt(observations)

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": summariser_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": 1000,
            "temperature": self.temperature,
        }
        return payload

    def get_gpt_response(self, observations) -> str:
        """
        parameters:
            observations (str): observations from agent's point of view

        return:
            filtered string of observations revelent to the task
        """
        payload = self.prepare_payload(observations)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        return self.parse_gpt_response(response, observations)

    def parse_gpt_response(self, response, observations):
        """
        parameters:
            response: gpt response object

        return:
            the filtered observations as a string
        """
        response_dict = response.json()

        try:
            content = response_dict["choices"][0]["message"]["content"]
            return str(content)
        except KeyError as e:
            raise KeyError(
                f"Key error: {e}. One of the keys is missing. obs {observations} response JSON {response_dict}"
            )
        except IndexError as e:
            raise IndexError(
                f"Index error: {e}. The list is empty or the index is out of range."
            )
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}.")

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

    objective_input = (
        "task: bring a tomato, lettuce and bread to the countertop to make a sandwich"
    )
    llm = ObsSummaryLLM(objective_input)

    observations = (
        "Cabinet_1, UpperCabinets_1, StandardCounterHeightWidth_1, Wall_1, Ceiling_1, "
        "Fridge_1, Toaster_1, CoffeeMachine_1, CounterTop_1"
    )

    response = llm.get_gpt_response(observations)
    print(response)
