"""
    General checker to check completion of subtasks.
    Also checks coverage of the task and success of the task.
"""

class BaseChecker:
    def __init__(
            self,
            subtasks,
            coverage,
            ) -> None:

        # NOTE: All subtasks are non-conditional for this environment
        self.subtasks = subtasks
        # objects or receptacles
        self.coverage = coverage

        self.subtasks_completed = []
        self.coverage_completed = []

        self.event=None

         # we also have self.available_object_names from the child class

    @property
    def subtasks_completed_numerated(self):
        # to store the subtasks completed with object number (@coela)
        # NOTE: This is the same as subtasks completed b/c we don't numerate
        return self.subtasks_completed

    def split_action(self, action: str):
        """Split action string into action and innards (can be tuple if multiple)"""
        act = action.split("(")[0]
        inner = action.split("(")[1].split(")")[0]
        if "," in inner:
            inner = tuple(inner.split(", "))
        return act, inner

    def unsplit_action(self, action: str, inner):
        if isinstance(inner, tuple):
            return f"{action}({', '.join(inner)})"
        return f"{action}({inner})"

    def give_credit_for_navigate(self, action, success):
        act, inner = self.split_action(action)
        if act not in ["NavigateTo"]:
            navigate_action = f"NavigateTo({object})"
            self._check_subtask(navigate_action, success)

    def _check_subtask(
        self,
        action,
        success,
    ):
        # idle actions - don't do anything
        if action in ["Done", "Idle"]:
            return None
        if "SendMessage" in action:
            return None

        # NOTE: replace the inner part w/ non-region (for fire)
        deregionize = lambda s : s[:s.index('_Region_')] if '_Region_' in s else s
        act, inner = self.split_action(action)
        if isinstance(inner, tuple):
            inner=tuple([deregionize(p) for p in inner])
        else: inner=deregionize(inner)
        # modified action after de-regionizing
        action = self.unsplit_action(act, inner)

        # if the action is in subtasks and not in subtasks_completed
        # and the action was successful, then add it to subtasks_completed
        if (
            action in self.subtasks
            and action not in self.subtasks_completed
            and success
        ):
            self.subtasks_completed.append(action)

            # Give credit for NavigateTo(object) if action(object) is successful
            # Since this means the agent just happened to already be close enough
            self.give_credit_for_navigate(action, success)

    def callback(self):
        """
        Callback function at the end of the fn.
        Implement in the child class.
        """
        raise NotImplementedError

    def get_transport_rate(self):
        return len(self.subtasks_completed) / len(self.subtasks)

    def check_coverage(self, action: str):
        for obj in self.coverage:
            if (obj in action) and (obj not in self.coverage_completed):
                self.coverage_completed.append(obj)

    def perform_metric_check(self, action, success, observation_dct):
        self._check_subtask(action, success)
        self.check_coverage(action)

        # do callback, can be inert
        self.event=observation_dct
        self.callback()

    def get_coverage(self):
        return len(self.coverage_completed) / len(self.coverage)

    def check_success(self):
        return len(self.subtasks_completed) == len(self.subtasks)
