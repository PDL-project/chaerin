from pathlib import Path
import collections
import json, os, sys
import glob
import pandas as pd
import numpy as np
import imageio
import re
import ast
import copy
from pprint import pprint

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

from misc import extract_number

class SARLogger:
    """
    Very similar to the AI2Thor logger

    Simple logger class to log:
        -the metrics in a csv file
        -the environment render images
    """

    def __init__(self, baseline_name, env=None):
        """
        Initialize
            -baseline_name -> name of the folder to put the log (name depending on baseline, i.e. 'ReAct')
            -env -> environment
            -summarize_mode -> True if you don't want to use log_step and only summarize currently logged data, False otherwise

        After one complete 'loop' update with the 'log_step' function below
            -log_step(step_num, action, action_successes, coverage, transport_rate)
        """

        self.summarize_mode=env is None

        self.env=env
        self.df_cols=['Step', 'Pre-Action', 'Action', 'Success', 'Coverage', 'Transport Rate', 'Finished']
        self.df = pd.DataFrame(columns=self.df_cols)
        self.inner_df_cols=['Step', 'Action', 'Reason', 'Subtask', 'Memory']
        self.inner_df = pd.DataFrame(columns=self.inner_df_cols)

        # make the save paths to be in baseline folder
        baseline_dir = Path(__file__).parent.absolute()
        self.baseline_path = Path(str(baseline_dir) + "/results/" + baseline_name)

        # self.baseline_path = Path("results") / baseline_name
        if not os.path.exists(self.baseline_path) and not self.summarize_mode:
            os.makedirs(self.baseline_path)

        if not self.summarize_mode:
            self.env.render_image_path = self.baseline_path / env.render_image_path
            if not os.path.exists(self.env.render_image_path):
                os.makedirs(self.env.render_image_path)

            # make sure we have the first frame in the right path (after we changed it)
            self.env.save_frame()

    def log_step(self, step, preaction, action, success, coverage, transport_rate, finished):
        """
        step  ->        current step in loop (i)
        preaction ->    action before it is semantically mapped
        action  ->          [agent_1_action, agent_2_action]
        action_successes -> [agent_1_action_success, agent_2_action_success]
        coverage   ->       env.checker.get_coverage()
        transport_rate  ->  env.checker.get_transport_rate()
        finished    ->      env.checker.check_success()
        """
        row = pd.DataFrame([[step, preaction, action, success, coverage, transport_rate, finished]], columns=self.df_cols)
        self.df = pd.concat([self.df, row])
        traj_path=self.baseline_path / self.env.action_dir_path
        if not os.path.exists(traj_path):
            os.makedirs(traj_path)
        self.df.to_csv(traj_path / str("trajectory.csv"), index=False)
    
    def recompute_metrics(self):
        traj_path_csv=self.baseline_path / self.env.action_dir_path / str("trajectory.csv")
        assert os.path.exists(traj_path_csv), f"File not found at {traj_path_csv} ... Cannot recompute statistics for nonexistent file"
        df = pd.read_csv(traj_path_csv)

        success_list = []
        coverage_list, transport_rate_list, finished_list = [], [], []
        for action_str in self.actions():
            # read in action from df
            action = ast.literal_eval(action_str)
            # execute action in environment
            d, action_successes = self.env.step(action)
            # recompute metrics
            coverage = self.env.checker.get_coverage()
            transport_rate = self.env.checker.get_transport_rate()
            finished = self.env.checker.check_success()
            # update lists
            success_list.append(action_successes)
            coverage_list.append(coverage)
            transport_rate_list.append(transport_rate)
            finished_list.append(finished)
        
        # update df with recomputed metrics
        df['Success'] = success_list
        df['Coverage'], df['Transport Rate'], df['Finished'] = coverage_list, transport_rate_list, finished_list
        # TODO: add multi-agent metrics recomputing to this fn
        self.df = df
        self.df.to_csv(traj_path_csv, index=False)

    def log_agent_mem(self, step, action, reason=None, subtask=None, memory=None):
        """
        Log the agent memory (baseline specific that it uses),
        most baselines don't require reason, subtask, or memory

        step  ->        current step on loop (i)
        action  ->      [agent_1_action, agent_2_action]
        reason ->       agent rationale for actions
        subtask ->      substasks agents should perform
        memory ->       agent memory for the subtask
        """

        # TODO: add all the results from the query to mem

        row = pd.DataFrame([[step,action,reason,subtask,memory]], columns=self.inner_df_cols)
        self.inner_df = pd.concat([self.inner_df, row])
        path=self.baseline_path / self.env.action_dir_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.inner_df.to_csv(path / str("memory.csv"), index=False)

    def actions(self):
        # get the agent actions from the current bath
        traj_path=self.baseline_path / self.env.action_dir_path
        if not os.path.exists(traj_path):
            os.makedirs(traj_path)
        pth = traj_path / str("trajectory.csv")
        df=pd.read_csv(pth)
        actns=df['Action'].to_numpy()
        actns=[ast.literal_eval(actn) for actn in actns]
        return actns

    def read_agent_mem(self):
        path=self.baseline_path / self.env.action_dir_path
        if not os.path.exists(path):
            os.makedirs(path)
        df=pd.read_csv(path / str("memory.csv"))
        return df

    def summarize(self):
        """
        Gives summary (average across different metrics) for the baseline_name folder
        You can feed in any env, it will give summary for the {num_agents} agent version (by default 2)

        Returns dictionary of dictionary.
            -first dict has keys ('task_string') and values (metric_dict)
                -each (metric_dict) has (keys) for metric type -> (values) list of specific metric for all floorplans
        
            Current metrics:
                -success_rate
                -transport_rate
                -coverage
                -average_steps
                -balance

        """
        # NOTE: (NEW) Gives list of dicts for all the n_agents encountered!

        sr=lambda df : int((df['Finished'].to_numpy())[-1])
        tr=lambda df : (df['Transport Rate'].to_numpy())[-1]
        c=lambda df : (df['Coverage'].to_numpy())[-1]
        st=lambda df : len(df['Step'].to_numpy())
        bl=lambda df: self._get_balance_metric(df['Action'].to_numpy(), df['Success'].to_numpy())
        
        self.metrics={'success_rate' : sr, 'transport_rate' : tr, 'coverage' : c, 'average_steps' : st, 'balance' : bl}

        # NOTE: slightly modified for the folder structure (actions/n_agents/seed_j/Scene_i/trajectory.csv)
        # seeds <-> floorplans, scenes <-> tasks, n_agents <-> different baselines altogether (use num_agents)

        n_agents=glob.glob(str( self.baseline_path / str("actions") / str("*") ))
        main_dicts={}

        n_agents.sort(key=lambda x : extract_number(Path(x).name))
        for n_agent in n_agents:
            main_dict={}

            make_scene_dict=lambda : dict([(k, []) for k in self.metrics.keys()])
            scene_dicts={} # maps scene_name -> it's scene_dict

            seeds=glob.glob(str(Path(n_agent) / str("*")))
            seeds.sort(key=lambda x : extract_number(Path(x).name))
            for seed in seeds:
                scenes=glob.glob(str(Path(seed) / str("*")))
                for scene in scenes:
                    pth=Path(scene) / str("trajectory.csv") # actions/n_agents/seed_j/Scene_i/trajectory.csv
                    df=pd.read_csv(pth)

                    scn_name=Path(scene).name
                    if scn_name not in scene_dicts.keys():
                        scene_dicts[scn_name]=make_scene_dict()
                        main_dict[scn_name]=scene_dicts[scn_name] # need to do once due to dict ptr

                    scn_dict=scene_dicts[scn_name]
                    for mn, mf in self.metrics.items():
                        scn_dict[mn].append(mf(df)) # add metrics to dict

            main_dicts[extract_number(Path(n_agent).name)]=main_dict

        return main_dicts

    # multiagent metric - balance
    def _get_balance_metric(self, actions, success):
        """
        Metric to get the balance in a certain episode.

        How it's measured:
        Let x_i be the ith agent amount of successful tasks completed.
        Then the metric is
        
        min(x_0,...,x_n)/max(x_0,...,x_n)

        This can be interpreted as the proportion of tasks the min agent does compared to the max agent.
        The useful properties is are:
            -For >2 agents, the metric properly captures the range of activity between agents (as opposed to metrics using averages).
                We can easily see disproportionate contributions
            -0 indicates no balance at all (one agent didn't do any tasks)
            -1 indicates perfect balance between all agents
        """
        def agentwise_actions(_filter=True):
            agent_actions=collections.defaultdict(list)
            for action_str, success_str in zip(actions, success):
                # action
                if '[' not in action_str and ']' not in action_str:
                    action_lst= [action_str]
                else:
                    action_lst = ast.literal_eval(action_str)

                # success
                if '[' not in action_str and ']' not in action_str:
                    success_lst=[success_str]
                else:
                    success_lst = ast.literal_eval(success_str)
                for agent_id in range(len(action_lst)):
                    agent_act=action_lst[agent_id]
                    success_act=success_lst[agent_id]
                    if _filter and agent_act in ["Done", "Idle"]:
                        continue
                    # only add successful actions
                    if success_act:
                        agent_actions[agent_id].append(agent_act)
            return agent_actions

        def metric(a_actions):
            # get lengths
            for k,v in a_actions.items():
                a_actions[k]=len(v)
            return min(a_actions.values()) / max(a_actions.values())

        return metric(agentwise_actions(_filter=True))

    def get_task_average(self, summary_dict):
        """
        Get the averages for each metric across floorplans (but keeps the task division intact)
        """
        s_dict=copy.deepcopy(summary_dict)
        for tsk,dct in s_dict.items():
            for metric,arr in dct.items():
                dct[metric]=np.array(arr).mean()
        return s_dict

    def get_overall_average(self, summary_dict):
        """
        Get the overall averages for each metric across all tasks and floorplans.
        NOTE: this calculates the overall average as two nested averages - first average across floorplans within a task, and THEN across tasks
            -this will give a different result than averaging all at once since not all tasks have the same # of floorplans
        """
        nd=dict([(k, []) for k in self.metrics.keys()])
        for tsk,dct in summary_dict.items():
            for metric,arr in dct.items():
                nd[metric].append(np.array(arr).mean())
        nd=dict([(k, np.array(v).mean()) for k,v in nd.items()])
        return nd
