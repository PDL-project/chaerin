from pathlib import Path
import collections
import json, os
import glob
import pandas as pd
import numpy as np
import re
import imageio
import ast
import copy
from pprint import pprint

class Logger:
    """
    Simple logger class to log:
        -the metrics in a csv file
        -the environment render images
    """

    def __init__(self, baseline_name, env=None, summarize_mode=False):
        """
        Initialize
            -baseline_name -> name of the folder to put the log (name depending on baseline, i.e. 'ReAct')
            -env -> environment
            -summarize_mode -> True if you don't want to use log_step and only summarize currently logged data, False otherwise

        After one complete 'loop' update with the 'log_step' function below
            -log_step(step_num, action, action_successes, coverage, transport_rate)
        """
        self.env=env
        self.df_cols=['Step', 'Pre-Action', 'Action', 'Success', 'Coverage', 'Transport Rate', 'Finished']
        self.df = pd.DataFrame(columns=self.df_cols)
        self.inner_df_cols=['Step', 'Action', 'Reason', 'Subtask', 'Memory']
        self.inner_df = pd.DataFrame(columns=self.inner_df_cols)

        # make the save paths to be in baseline folder
        baseline_dir = Path(__file__).parent.parent.absolute()
        self.baseline_path = Path(str(baseline_dir) + "/results/" + baseline_name)

        # self.baseline_path = Path("results") / baseline_name
        if not os.path.exists(self.baseline_path) and not summarize_mode:
            os.makedirs(self.baseline_path)

        self.summarize_mode=summarize_mode
        if not summarize_mode:
            self.env.render_image_path = self.baseline_path / env.render_image_path
            if not os.path.exists(self.env.render_image_path):
                os.makedirs(self.env.render_image_path)

            # make sure we have the first frame in the right path (after we changed it)
            for agent_id in range(env.num_agents):
                self.env.save_frame(agent_id)

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
        row = pd.DataFrame([[step,action,reason,subtask,memory]], columns=self.inner_df_cols)
        self.inner_df = pd.concat([self.inner_df, row])
        path=self.baseline_path / self.env.action_dir_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.inner_df.to_csv(path / str("memory.csv"), index=False)

    def actions(self, num_agents=2):
        # get the agent actions from the current bath
        traj_path=self.baseline_path / self.env.action_dir_path
        if not os.path.exists(traj_path):
            os.makedirs(traj_path)
        pth = traj_path / str("trajectory.csv")
        df=pd.read_csv(pth)
        actns=df['Action'].to_numpy()
        return actns

    def recreate(self, actions, path=None, overhead=True):
        """
        Recreates images from action list (of lists) as a gif,
        actions -> list of lists w/ actions for each agent
        path -> optional location of file to put gif
        overhead -> flag for whether to add overhead images too
        """
        # Appends path  to current render image path
        if path is None:
            path=self.env.render_image_path
        else:
            path= Path(path)

        # get the directory of the agents
        extract_number=lambda s: int(re.findall(r'\d+',s)[0])
        def get_agent_dirs():
            name = lambda p : Path(p).name
            agent_dirs=glob.glob(str(path / "*"))
            agent_dirs=list(filter(lambda f : name(f) in self.env.agent_names, agent_dirs)) # no files, only folders, not 'gifs' folder
            agent_dirs.sort(key=lambda f : name(f))
            # create agent directories!
            for ad in agent_dirs:
                nm=name(ad)
                if not os.path.exists(path / nm):
                    os.mkdir(path / nm)
            return agent_dirs

        agent_dirs=get_agent_dirs()

        path=path / 'gifs'
        if not os.path.exists(path):
            os.makedirs(path)
        tmp=self.env.render_image_path
        self.env.render_image_path=path
        self.env.set_overhead(overhead)

        print('path w/ gif:', path)

        # get the filenames currently in the folder (within folder mode)
        def filenames(agent_dirs=None, lmt=None, mode='pov'):
            if agent_dirs is None:
                agent_dirs=get_agent_dirs()
            agent_filenames=[]
            for i,agent in enumerate(agent_dirs):
                name=Path(agent).name
                filenames=glob.glob(str(path / (name) / (mode) / ("*")))
                filenames.sort(key=lambda f : extract_number(Path(f).name.split('.')[0]))
                if lmt is not None:
                    filenames=filenames[:lmt]
                agent_filenames.append(filenames)
            return agent_filenames

        # if overhead, do global filenames and merge agent's 1,2,...,n frames together
        if overhead:
            mem=collections.defaultdict(list)
            global_filenames=[]
            idx=0

        for action in actions:
            self.env.step(action)

            # save frames
            for agent_id in range(self.env.num_agents):
                self.env.save_frame(agent_id)
                # limit frames
                snh=self.env.step_nums_history[self.env.agent_names[agent_id]]
                sn_mem=snh[-2] if len(snh)>1 else 0
                diff=snh[-1]-sn_mem

                # do all this so that we have a 'global clock'
                if overhead:
                    # uses the fact that agent names are in lexographical order
                    _agent_dir=[agent_dirs[agent_id]]
                    curr_fns=filenames(agent_dirs=_agent_dir, lmt=diff, mode='overhead')[0]
                    old_fns=mem[agent_id]
                    new_fns=[f for f in curr_fns if f not in old_fns]
                    global_filenames+=new_fns
                    mem[agent_id]=curr_fns

        # regardless of overhead flag, create the gif for each individual agent (we get it for free)
        agent_filenames=filenames()
        for agent,filenames in zip(agent_dirs,agent_filenames):
            name=Path(agent).name
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
            if images:
                print(f'saving local images to {name}.gif')
                imageio.mimsave(path / str(f'{name}.gif'), images)

        if overhead:
            global_images=[]
            for filename in global_filenames:
                global_images.append(imageio.imread(filename))
            if global_images:
                print(f'saving global images to global.gif')
                imageio.mimsave(path / str(f'global.gif'), global_images)

        # go back to original
        self.env.render_image_path=tmp

    def read_agent_mem(self):
        path=self.baseline_path / self.env.action_dir_path
        if not os.path.exists(path):
            os.makedirs(path)
        df=pd.read_csv(path / str("memory.csv"))
        return df

    def summarize_matrix(self, num_agents=2, max_steps=30, zero_initial=True, as_dict=True):
        # TODO: more dynamic changes in max_steps, assert division between tasks w/ diff timeout number
        """
        Give summary of all the metrics in the form of a 3-d array
        metrics (3) x datapoints (amt of task-floorplan pairs) x (max_steps+int(zero_initial))

        metrics (others don't make sense in this context):
        -success rate
        -transport_rate
        -coverage

        *in this order

        Useful for visualization of metric across time w/ ci
        """
        
        sr=lambda df : np.array([int(v) for v in df['Finished'].to_numpy()])
        tr=lambda df : df['Transport Rate'].to_numpy()
        c=lambda df : df['Coverage'].to_numpy()
        st=lambda df : len(df['Step'].to_numpy())

        def pad(a, width):
            # pad 1-d np array by extending last value until size
            return np.pad(a, width, mode='edge')[width:]
        
        NUM_METRICS=3
        metrics={'success_rate' : sr, 'transport_rate' : tr, 'coverage' : c}

        extract_number=lambda s: int(re.findall(r'\d+',s)[0])
        tsk_dirs=glob.glob(str( self.baseline_path / str("actions") / str("*") ))

        tsk_fp_mt_arr=[]
        for tsk_dir in tsk_dirs:
            floorplans=glob.glob(str( Path(tsk_dir) / str("*") ))
            floorplans.sort(key=lambda x : extract_number(Path(x).name))
            for fp in floorplans:
                pth=Path(fp) / str(num_agents) / str("trajectory.csv")
                df=pd.read_csv(pth)

                fp_arr=[]
                # have df in tsk_dir x fp
                for mn, mf in metrics.items():
                    l=mf(df)
                    l=pad(l, max_steps-len(l))
                    if zero_initial:
                        l=np.insert(l,0,0)
                    fp_arr.append(l)

                # fp_arr - metrics x time
                tsk_fp_mt_arr.append(fp_arr)
        # tsk_fp_mt_arr - (tsk * fp) x metrics x time
        tsk_fp_mt_arr=np.array(tsk_fp_mt_arr)

        # -1 x metrics x time
        mat=tsk_fp_mt_arr.transpose((1,0,2)) # permute -1 x metrics to get metrics x -1 x time

        if as_dict:
            # return as dict
            d={}
            for i, nm in enumerate(metrics.keys()):
                d[nm]=mat[i] # index the metric
            return d

        return mat

    def summarize(self, num_agents=2):
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
        sr=lambda df : int((df['Finished'].to_numpy())[-1])
        tr=lambda df : (df['Transport Rate'].to_numpy())[-1]
        c=lambda df : (df['Coverage'].to_numpy())[-1]
        st=lambda df : len(df['Step'].to_numpy())
        bl=lambda df: self._get_balance_metric(df['Action'].to_numpy(), df['Success'].to_numpy())
        
        self.metrics={'success_rate' : sr, 'transport_rate' : tr, 'coverage' : c, 'average_steps' : st, 'balance' : bl}
        main_dict={}

        extract_number=lambda s: int(re.findall(r'\d+',s)[0])
        tsk_dirs=glob.glob(str( self.baseline_path / str("actions") / str("*") ))
        for tsk_dir in tsk_dirs:
            tsk_dict=dict([(k, []) for k in self.metrics.keys()])
            floorplans=glob.glob(str( Path(tsk_dir) / str("*") ))
            floorplans.sort(key=lambda x : extract_number(Path(x).name))
            for fp in floorplans:
                pth=Path(fp) / str(num_agents) / str("trajectory.csv")
                df=pd.read_csv(pth)
                for mn, mf in self.metrics.items():
                    tsk_dict[mn].append(mf(df))
            tsk_name=Path(tsk_dir).name
            main_dict[tsk_name]=tsk_dict

        return main_dict

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
