"""LLM for summarising available actions in the history for the AI2Thor environment"""
import openai
from openai.error import RateLimitError, APIConnectionError
from time import sleep

act_summariser_prompt = """You are an excellent decision-maker for an embodied robot. You are helping another human being in carrying out a task. The human being will give you a list of objects that it is observing and the task it is trying to carry out. You need to give a list of possible actions that could be taken by the human at the current timestep to carry out the task efficiently.
The <object> and <receptacle> in the observation are of the form "<object_name>_<object_id>".
Note that you need to choose actions from the following list of actions:
["Move(direction)", "Rotate(rotation)", "OpenObject(object)", "CloseObject(object)", "PickupObject(object)", "PutObject(receptacle)"]
Here <direction> can be one of ["Ahead", "Back", "Left", "Right"], and <rotation> can be one of ["Right", "Left"].
Note that <object> and <receptacle> can only be from what the agent sees in "<agent_name>'s Observation". In essence, you can only afford to interact with objects/receptacles that the agent can see.

Input format: {Task Description: description of the task the human is trying to achieve, Observation: list of objects that the human is observing in the current time step, Agent State: description of agent's state in the current time step}
Output format: List of relevant actions that could be carried out in the environment in the current time step.

Here is are a few examples of input and output and explanations for why some actions are not included in the output:
Example 1:
Input: {Task Description: "bring a tomato, lettuce and bread to the countertop to make a sandwich", Observation: ["Cabinet_1", "UpperCabinets_1", "Fridge_1", "Toaster_1", "CoffeeMachine_1", "CounterTop_1"], Agent State: "I am at co-ordinates: (0.5, 0.9, -1.25) and I am holding nothing"}
Output: ["Move(Ahead)", "Move(Back)", "Move(Right)", "Move(Left)", "Rotate(Right)", "Rotate(Left)", "OpenObject(Cabinet_1)", "OpenObject(UpperCabinets_1)", "OpenObject(Fridge_1)", "CloseObject(Cabinet_1)", "CloseObject(UpperCabinets_1)", "CloseObject(Fridge_1)"]
Wrong Output: ["PickupObject(Lettuce_1)", "CloseObject(CoffeeMachine_1)", "PutObject(CounterTop_1)"]
Explanation for why each action in "Wrong Output" is wrong: 
    "PickupObject(Lettuce_1)": The agent does not see "Lettuce_1" in its observation, so it cannot pick it up.
    "CloseObject(CoffeeMachine_1)": It doesn't make sense to close a coffee machine.
    "PutObject(CounterTop_1)": The agent is not holding anything, so it cannot put anything down on the "CounterTop_1".

Example 2:
Input: {Task Description: "bring a tomato, lettuce and bread to the countertop to make a sandwich", Observation: ["Cabinet_1", "Fridge_1", "Tomato_1", "CounterTop_1"], Agent State: "I am at co-ordinates: (0.75, 0.9, -2.0) and I am holding nothing"}
Output: ["Move(Ahead)", "Move(Back)", "Move(Right)", "Move(Left)", "Rotate(Right)", "Rotate(Left)", "OpenObject(Cabinet_1)", "OpenObject(Fridge_1)", "CloseObject(Cabinet_1)", "CloseObject(Fridge_1)", "PickupObject(Tomato_1)"]
Wrong Output: ["PickupObject(Lettuce_1)", "OpenObject(CoffeeMachine_1)", "PutObject(Tomato_1)"]
Explanation for why each action in "Wrong Output" is wrong:
    "PickupObject(Lettuce_1)": The agent does not see "Lettuce_1" in its observation, so it cannot pick it up.
    "OpenObject(CoffeeMachine_1)": The agent does not see "CoffeeMachine_1" in its observation, so it cannot open it.
    "PutObject(Tomato_1)": It doesn't make sense to put anything on top of "Tomato_1". Also, the agent is not holding anything.

Example 3:
Input: {Task Description: "bring a tomato, lettuce and bread to the countertop to make a sandwich", Observation: ["Cabinet_1", "Fridge_1", "Tomato_1", "CounterTop_1"], Agent State: "I am at co-ordinates: (0.75, 0.9, -2.0) and I am holding Tomato_1"}
Output: ["Move(Ahead)", "Move(Back)", "Move(Right)", "Move(Left)", "Rotate(Right)", "Rotate(Left)", "OpenObject(Cabinet_1)", "OpenObject(Fridge_1)", "CloseObject(Cabinet_1)", "CloseObject(Fridge_1)", "PutObject(CounterTop_1)"]
Wrong Output: ["PickupObject(Tomato_1)", "PutObject(Wall_1)"]
Explanation for why each action in "Wrong Output" is wrong:
    "PickupObject(Tomato_1)": The agent is able to see "Tomato_1" in its observation, but it is already holding it so it cannot pick it up again.
    "PutObject(Wall_1)": It doesn't make sense to put "Tomato_1" on "Wall_1".
"""

import json
import openai
from openai.error import RateLimitError, APIConnectionError
from time import sleep

import os, sys

sys.path.append(os.path.abspath((os.getcwd())))

from AI2Thor.strict_output import strict_output


class ActSummaryLLM(object):
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
            act_summariser_prompt,
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
    llm = ActSummaryLLM(llm_cache, model)
    user_prompt = {
        "Task Description": "bring a tomato, lettuce and bread to the countertop to make a sandwich",
        "Observation": ["Cabinet_1", "Fridge_1", "Tomato_1", "CounterTop_1"],
        "Agent State": "I am at co-ordinates: (0.75, 0.9, -2.0) and I am holding Tomato_1",
    }
    # should return ["Cabinet_1", "UpperCabinets_1", "Fridge_1", "CounterTop_1"]
    output_format = {"Action": ["PickupObject(Tomato_1)", "PutObject(Wall_1)"]}
    response = llm.get_gpt_response(user_prompt, output_format)
    print(response[0])
    print(response[0][0])
    print(len(response[0]))
    print(response[0][5])
