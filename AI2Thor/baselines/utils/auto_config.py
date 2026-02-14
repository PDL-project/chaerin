from pathlib import Path
import json, os
import glob


class BaseConfig:
    def __init__(self):
        self.num_agents = 2
        self.scene = "FloorPlan1"
        self.scene_name = "FloorPlan1"
        self.model = "gpt-4"

        self.use_langchain = False
        self.use_strict_format = True
        self.use_obs_summariser = False
        self.use_act_summariser = False
        self.use_action_failure = True
        self.use_shared_subtask = True
        self.use_separate_subtask = False
        self.use_future_message = True
        self.forceAction = False
        self.use_memory = True
        self.use_plan = True
        self.use_separate_memory = False
        self.use_shared_memory = True
        self.temperature = 0.7


class AutoConfig:
    """
    AutoConfig class to automatically parse the config.json file.
    It can create a config object given 'task index' given to use in env,
    it can give the task_string for the current task index,
    and it can give the task timeout number as well

    config_file -> configuration file path ('config.json' by default)
    base_config -> base configuration parameters ('BaseConfig' above by default)

    ------------------------

    *use config class as such:
    '''
    auto=AutoConfig()
    # set current to task to be the one at index '0' and floorplan index '0'
    auto.set_task(0)
    auto.set_floorplan(0)

    env = AI2ThorEnv(auto.config())
    task = auto.task_string()
    d = env.reset(task=task)
    '''

    *to loop through all the tasks and all floorplans, use the 'get_amt_tasks()' and 'get_amt_floorplans(task_index)' function:
    '''
    from baseline_utils.auto_config import AutoConfig

    auto=AutoConfig()
    amt_tasks=auto.get_amt_tasks()
    for task_index in range(amt_tasks):

        auto.set_task(task_index) # set task index
        amt_floorplans=auto.get_amt_floorplans(task_index)

        for fp_index in range(amt_floorplans):

            auto.set_floorplan(fp_index) # set floorplan
            timeout=auto.get_task_timeout() # get timeout number

            env = AI2ThorEnv(auto.config())
            task = auto.task_string()
            d = env.reset(task=task)
            # do things ...
    '''

    When using it ALWAYS set the 'task_index' and 'floorplan_index'

    """

    def __init__(self, config_file="config.json", base_config=None, verbose=True):
        self.base_config = base_config
        self.verbose = verbose
        if base_config is None:
            self.base_config = BaseConfig()
        
        # Construct the absolute path to the config file to address relative imports
        config_dir = Path(__file__).parent.parent.parent.parent.absolute()
        config_path = config_dir / config_file

        with open(config_path, "r") as f:
            self.config_dict = json.load(f)
        self.config_arr = self.config_dict["tasks"]

        self.task_index = None
        self.floorplan_index = None
        self.agents=2 # default to 2

    def set_agents(self, agents):
        self.agents=agents

    def get_amt_tasks(self):
        """
        Get amt of tasks in the config file (useful for iterating through them)
        """
        return len(self.config_arr)

    def get_amt_floorplans(self, task_index):
        if task_index is None:
            task_index = self.task_index
        """
        Get amt of floorplans for a specific 'task_index' the config file (useful for iterating through them)
        """
        assert self.task_index is not None, "Task index has not been specified yet."
        assert self._within_bounds_tsk(
            task_index
        ), f"Task index {task_index} not within bounds. Min: 0, Max: {self.get_amt_tasks()}"
        return len(self.config_arr[task_index]["task_floorplans"])

    def _within_bounds_tsk(self, tsk_idx):
        return 0 <= tsk_idx < self.get_amt_tasks()

    def _within_bounds_fp(self, fp_idx):
        assert self.task_index is not None, "Task index has not been specified yet."
        return 0 <= fp_idx < self.get_amt_floorplans(self.task_index)

    def set_task(self, task_index):
        """
        Set task to specific index
        """
        assert self._within_bounds_tsk(
            task_index
        ), f"Task index {task_index} not within bounds. Min: 0, Max: {self.get_amt_tasks()}"
        self.task_index = task_index
        self.floorplan_index = None
        task_name = self.task_string()  # self.config_arr[task_index]['task_name']
        if self.verbose:
            print(
                f"Setting Config Task to {task_index}, with task name '{task_name}'. Floorplan unknown."
            )

    def _get_floorplan_name(self, floorplan_index):
        assert self._within_bounds_fp(
            floorplan_index
        ), f"Floorplan index {floorplan_index} not within bounds. Min: 0, Max: {self.get_amt_floorplans(self.task_index)}"
        return self.config_arr[self.task_index]["task_floorplans"][floorplan_index]

    def _get_task_name(self, task_index):
        assert self._within_bounds_tsk(
            task_index
        ), f"Task index {task_index} not within bounds. Min: 0, Max: {self.get_amt_tasks()}"
        return self.config_arr[self.task_index]["task_description"]

    def set_floorplan(self, floorplan_index):
        """
        Set floorplan to specific index
        """
        assert self.task_index is not None, "Task index has not been specified yet."
        assert self._within_bounds_fp(
            floorplan_index
        ), f"Floorplan index {floorplan_index} not within bounds. Min: 0, Max: {self.get_amt_floorplans(self.task_index)}"
        floorplan_name = self._get_floorplan_name(floorplan_index)
        if self.verbose:
            print(
                f"Setting Config FloorPlan to {floorplan_index}, with name '{floorplan_name}'"
            )
        self.floorplan_index = floorplan_index

    def get_task_timeout(self):
        assert self.task_index is not None, "Task index has not been specified yet."
        return self.config_arr[self.task_index]["task_timeout"]

    def task_string(self):
        return self._get_task_name(self.task_index)

    def config(self):
        """
        Create configuration file
        """
        assert (
            self.task_index is not None and self.floorplan_index is not None
        ), "Either task, floorplan, or both haven't been specified"
        cfg = self.base_config
        cfg.scene = self._get_floorplan_name(self.floorplan_index)
        cfg.scene_name = cfg.scene
        cfg.num_agents = self.agents
        return cfg
