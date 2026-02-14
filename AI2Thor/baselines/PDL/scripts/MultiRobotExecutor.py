#!/usr/bin/env python3
"""
MultiRobotExecutor - Run multi-robot plan in AI2-THOR directly.
Based on the working ai2_thor_controller.py and aithor_connect.py patterns.
"""

import json
import math
import os
import re
import threading
import time
import random
import shutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
from glob import glob
from pathlib import Path
import sys

# Ensure repo root is on sys.path for AI2Thor imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from ai2thor.controller import Controller
from scipy.spatial import distance
from AI2Thor.Tasks.get_scene_init import get_scene_initializer


# -----------------------------
# 각 서브태스크의 ID, 이름, 담당 로봇 ID, 수행할 액션 리스트, 병렬 그룹 번호를 저장하는 데이터 구조
# -----------------------------
@dataclass
class SubtaskPlan:
    subtask_id: int
    subtask_name: str
    robot_id: int               # 1-based
    actions: List[str]
    parallel_group: int = 0

@dataclass
class YieldRequest:
    requester_id: int
    timestamp: float
    reason: str
    target_object: str
    attempts: int = 0
    last_distance: float = 0.0
    next_time: float = 0.0

# -----------------------------
# 이동
# -----------------------------
def closest_node(node, nodes, no_robot, clost_node_location):
    """로봇이 특정 목적지까지 갈 때, 시뮬레이션 내에서 이동 가능한(Reachable) 가장 가까운 지점을 계산하는 함수"""
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append(nodes[pos_index])
    return crps


def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    """두 지점 사이의 2차원 평면(x, z) 거리를 계산하는 함수"""
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

# -----------------------------
# 실행기
# -----------------------------
class MultiRobotExecutor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.resources_path = os.path.join(base_path, "resources")
        self.dag_output_path = os.path.join(self.resources_path, "dag_outputs")
        self.plans_path = os.path.join(self.resources_path, "subtask_pddl_plans")

        self.assignment: Dict[int, int] = {}
        self.parallel_groups: Dict[int, List[int]] = {}
        self.subtask_plans: Dict[int, SubtaskPlan] = {}

        # AI2-THOR
        self.controller: Optional[Controller] = None
        self.no_robot = 1
        self.reachable_positions: List[Tuple[float, float, float]] = []
        self.reachable_positions_: List[dict] = []

        self.action_queues: List[deque] = []
        self.action_lock = threading.Lock()
        self.rr_index = 0
        self.task_over = False
        self.actions_thread: Optional[threading.Thread] = None

        # Stats
        self.total_exec = 0
        self.success_exec = 0

        # Checker
        self.checker = None
        self.object_dict: Dict[str, Dict[str, int]] = {}
        self.inventory: List[str] = []
        self.scene_name: Optional[str] = None

        self.bb_lock = threading.Lock()
        self.yield_requests: Dict[int, YieldRequest] = {}  # key: blocking_agent_id(0-based)
        self.monitor_thread: Optional[threading.Thread] = None
        self.yield_cooldown_s = 3.0
        self.last_yield_request: Dict[Tuple[int, int], float] = {}  # (blocking_id, requester_id) -> timestamp
        self.yield_clear_distance = 1.5
        self.yield_margin = 0.3
        self.yield_retry_delay_s = 0.5

        # Action completion signaling (per-agent)
        self.action_cv = threading.Condition()
        self.agent_action_counters: List[int] = []

        # Receptacle cache (agent_id, pattern) -> objectId
        self.receptacle_cache: Dict[Tuple[int, str], str] = {}

        # NAV rotation-only detection
        self.nav_rotation_only_count: List[int] = []
        self.nav_rotation_only_threshold = 6

        # NAV oscillation (forward-backward) detection
        self.nav_position_history_size = 12  # 최근 N개 위치 저장 (실제 이동한 위치만)
        self.nav_oscillation_radius = 0.25   # 이 반경 이내면 "같은 위치"로 판단
        self.nav_oscillation_threshold = 4   # N회 재방문 감지 시 oscillation으로 판정
        self.nav_oscillation_move_thresh = 0.15  # 이 거리 이상 움직여야 히스토리에 기록
        self.nav_oscillation_cooldown_iters = 10  # recovery 후 N회 반복동안 감지 건너뜀

    # -----------------------------
    # Checker helpers (SmartLLM-style)
    # -----------------------------
    def _parse_object(self, object_str: str):
        obj_name = object_str.split("|")[0]
        obj_id = object_str.replace(obj_name, "")
        return obj_name, obj_id

    def _build_object_dict(self):
        self.object_dict = {}
        for obj in self.controller.last_event.metadata["objects"]:
            obj_name, obj_id = self._parse_object(obj["objectId"])
            if obj_name not in self.object_dict:
                self.object_dict[obj_name] = {}
            if obj_id not in self.object_dict[obj_name]:
                self.object_dict[obj_name][obj_id] = len(self.object_dict[obj_name]) + 1

    def _convert_object_id_to_readable(self, object_id: str) -> str:
        if not object_id:
            return "unknown"
        if "|" not in object_id:
            # try to resolve a raw name to an objectId
            resolved = self._find_object_id(object_id)
            if not resolved:
                return object_id
            object_id = resolved
        obj_name, obj_id = self._parse_object(object_id)
        if obj_name not in self.object_dict or obj_id not in self.object_dict[obj_name]:
            return obj_name
        obj_num = self.object_dict[obj_name][obj_id]
        return f"{obj_name}_{obj_num}"

    def _update_inventory(self, agent_id: int):
        inv = self.controller.last_event.events[agent_id].metadata.get("inventoryObjects", [])
        if not inv:
            self.inventory[agent_id] = "nothing"
            return
        # take first item if multiple
        self.inventory[agent_id] = self._convert_object_id_to_readable(inv[0]["objectId"])

    def _init_checker(self, task_description: str, scene_name: str):
        scene_initializer, checker_mod = get_scene_initializer(task_description, scene_name)
        self.checker = checker_mod.Checker()
        # ensure coverage uses current scene objects
        all_oids = [obj["objectId"] for obj in self.controller.last_event.metadata["objects"]]
        if hasattr(self.checker, "all_objects"):
            self.checker.all_objects(obj_ids=all_oids, scene=scene_name)
        print("_" * 50)
        print("Subtasks to complete:")
        try:
            print("\n".join(self.checker.subtasks))
        except Exception:
            pass
        if scene_initializer is not None:
            self.controller.last_event = scene_initializer.SceneInitializer().preinit(
                self.controller.last_event, self.controller
            )

    def _enqueue_front(self, actions: List[dict]):
        # actions should include agent_id
        with self.action_lock:
            for action in reversed(actions):
                self._enqueue_action_locked(action, front=True)

    def _enqueue_action_locked(self, action: dict, front: bool = False):
        agent_id = int(action.get("agent_id", 0))
        if not self.action_queues:
            return
        if agent_id < 0 or agent_id >= len(self.action_queues):
            agent_id = 0
        if front:
            self.action_queues[agent_id].appendleft(action)
        else:
            self.action_queues[agent_id].append(action)

    def _enqueue_action(self, action: dict, front: bool = False):
        with self.action_lock:
            self._enqueue_action_locked(action, front=front)

    def _queue_total_len(self) -> int:
        with self.action_lock:
            return sum(len(q) for q in self.action_queues)

    def _dequeue_action(self) -> Optional[dict]:
        with self.action_lock:
            if not self.action_queues:
                return None
            n = len(self.action_queues)
            for i in range(n):
                idx = (self.rr_index + i) % n
                if self.action_queues[idx]:
                    act = self.action_queues[idx].popleft()
                    self.rr_index = (idx + 1) % n
                    return act
        return None

    def _enqueue_and_wait(self, action: dict, agent_id: int, timeout: float = 10.0) -> bool:
        """Enqueue a single action and wait until it is processed for the given agent."""
        with self.action_cv:
            if not self.agent_action_counters:
                self._enqueue_action(action)
                return True
            start = self.agent_action_counters[agent_id]
            self._enqueue_action(action)
            end_time = time.time() + timeout
            while self.agent_action_counters[agent_id] == start:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return False
                self.action_cv.wait(timeout=remaining)
            return True

    # -----------------------------
    # 데이터 로드
    # -----------------------------
    def load_assignment(self, task_idx: int = 0) -> Dict[int, int]:
        """"어떤 서브태스크를 어떤 로봇이 맡을지" 기록된 JSON 파일 읽어오는 함수"""
        assignment_file = os.path.join(self.dag_output_path, f"task_{task_idx}_assignment.json")
        if not os.path.exists(assignment_file):
            raise FileNotFoundError(f"Assignment file not found: {assignment_file}")

        with open(assignment_file, "r") as f:
            data = json.load(f)

        self.assignment = {int(k): int(v) for k, v in data.get("assignment", {}).items()}

        # 전체 에이전트 수 로드 (할당 안 된 로봇도 소환하기 위해)
        self.configured_agent_count = data.get("agent_count", None)

        # LP에서 결정된 스폰 좌표 로드 (실행 시 동일 위치 재배치용)
        raw_spawn = data.get("robot_spawn_positions", None)
        if raw_spawn:
            self.saved_spawn_positions = {int(k): tuple(v) for k, v in raw_spawn.items()}
        else:
            self.saved_spawn_positions = None

        print(f"[Executor] Loaded assignment: {self.assignment}")
        if self.configured_agent_count:
            print(f"[Executor] Configured agent count: {self.configured_agent_count}")
        return self.assignment

    def load_subtask_dag(self, task_name: str = "task") -> Dict[int, List[int]]:
        """"어떤 서브태스크들이 동시에 실행 가능한지(Parallel Groups)" 기록된 DAG 결과 파일 읽어오는 함수"""
        dag_file = os.path.join(self.dag_output_path, f"{task_name}_SUBTASK_DAG.json")
        if not os.path.exists(dag_file):
            raise FileNotFoundError(f"Subtask DAG file not found: {dag_file}")

        with open(dag_file, "r") as f:
            data = json.load(f)

        raw_pg = data.get("parallel_groups", {})
        self.parallel_groups = {int(k): list(v) for k, v in raw_pg.items()}
        print(f"[Executor] Loaded parallel groups: {self.parallel_groups}")
        return self.parallel_groups

    def load_plan_actions(self) -> Dict[int, List[str]]:
        """각 서브태스크별로 수행해야 할 구체적인 PDDL 액션 리스트(_actions.txt)를 읽어와 SubtaskPlan 객체로 만드는 함수"""
        if not os.path.exists(self.plans_path):
            raise FileNotFoundError(f"Plans directory not found: {self.plans_path}")

        plan_files = [f for f in os.listdir(self.plans_path) if f.endswith("_actions.txt")]
        plan_actions: Dict[int, List[str]] = {}

        for plan_file in sorted(plan_files):
            m = re.search(r"subtask_(\d+)", plan_file)
            if not m:
                continue
            subtask_id = int(m.group(1))

            plan_path = os.path.join(self.plans_path, plan_file)
            with open(plan_path, "r") as f:
                actions = [ln.strip() for ln in f.readlines() if ln.strip()]

            plan_actions[subtask_id] = actions
            subtask_name = plan_file.replace("_actions.txt", "")

            robot_id = int(self.assignment.get(subtask_id, 1))
            pg = 0
            for gid, sids in self.parallel_groups.items():
                if subtask_id in sids:
                    pg = gid
                    break

            self.subtask_plans[subtask_id] = SubtaskPlan(
                subtask_id=subtask_id,
                subtask_name=subtask_name,
                robot_id=robot_id,
                actions=actions,
                parallel_group=pg,
            )

        print(f"[Executor] Loaded {len(plan_actions)} subtask plans")
        return plan_actions

    # -----------------------------
    # Action Queue Executor 액션 실행기
    # -----------------------------
    def _exec_actions(self):
        """백그라운드에서 action_queue에 쌓인 명령들을 하나씩 꺼내 시뮬레이터(controller.step)에 전달하고, 화면(OpenCV 창)을 갱신"""
        c = self.controller
        img_counter = 0

        while not self.task_over:
            act = self._dequeue_action()
            if act is not None:
                try:
                    multi_agent_event = None

                    if act['action'] == 'ObjectNavExpertAction':
                        multi_agent_event = c.step(dict(
                            action=act['action'],
                            position=act['position'],
                            agentId=act['agent_id']
                        ))
                        # actionReturn을 두 곳에서 모두 확인 (AI2Thor 버전 호환성)
                        aid = act['agent_id']
                        next_action = None
                        # 1) 해당 에이전트의 per-agent metadata
                        try:
                            next_action = multi_agent_event.events[aid].metadata.get('actionReturn')
                        except Exception:
                            pass
                        # 2) fallback: 글로벌 metadata (단일 에이전트 호환)
                        if next_action is None:
                            next_action = multi_agent_event.metadata.get('actionReturn')

                        # 디버그: 처음 5번만 로그
                        if img_counter < 5:
                            success = multi_agent_event.events[aid].metadata.get('lastActionSuccess', '?')
                            err = multi_agent_event.events[aid].metadata.get('errorMessage', '')
                            print(f"[NAV DEBUG] agent={aid}, actionReturn={next_action}, success={success}, err={err}")

                        # (A) actionReturn이 문자열이면 그대로 action으로 실행
                        if isinstance(next_action, str) and next_action:
                            multi_agent_event = c.step(
                                action=next_action,
                                agentId=aid,
                            )
                        # (B) actionReturn이 dict면 파라미터 포함하여 실행
                        elif isinstance(next_action, dict) and next_action.get("action"):
                            cmd = dict(next_action)
                            cmd["agentId"] = aid
                            multi_agent_event = c.step(cmd)
                        # (C) None이면 아무것도 안 함

                        # blocking 감지: errorMessage에 "blocking"이 있으면 회피 유도
                        try:
                            err = multi_agent_event.events[aid].metadata.get("errorMessage", "") or ""
                            if "blocking" in err.lower():
                                self._enqueue_action({
                                    'action': 'MoveBack',
                                    'agent_id': aid
                                }, front=True)
                        except Exception:
                            pass

                        # rotation-only detection (문자열 + dict 모두 처리)
                        try:
                            act_name = None
                            if isinstance(next_action, str):
                                act_name = next_action
                            elif isinstance(next_action, dict):
                                act_name = next_action.get("action")

                            if act_name in ("RotateLeft", "RotateRight"):
                                if aid < len(self.nav_rotation_only_count):
                                    self.nav_rotation_only_count[aid] += 1
                            else:
                                if aid < len(self.nav_rotation_only_count):
                                    self.nav_rotation_only_count[aid] = 0
                        except Exception:
                            pass

                    elif act['action'] == 'Teleport':
                        multi_agent_event = c.step(dict(
                            action="Teleport",
                            position=act['position'],
                            agentId=act['agent_id'],
                            forceAction=True
                        ))

                    elif act['action'] == 'MoveAhead':
                        multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])

                    elif act['action'] == 'MoveBack':
                        multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])

                    elif act['action'] == 'RotateLeft':
                        multi_agent_event = c.step(
                            action="RotateLeft",
                            degrees=act['degrees'],
                            agentId=act['agent_id']
                        )

                    elif act['action'] == 'RotateRight':
                        multi_agent_event = c.step(
                            action="RotateRight",
                            degrees=act['degrees'],
                            agentId=act['agent_id']
                        )

                    elif act['action'] == 'PickupObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="PickupObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[PickupObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'PutObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="PutObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        err_msg = multi_agent_event.metadata.get('errorMessage', "")
                        if err_msg != "":
                            print(f"[PutObject] Error: {err_msg}")
                            # Auto-recovery: if receptacle closed, open then retry once
                            if "CLOSED" in err_msg.upper():
                                retry = act.get("retry", 0)
                                if retry < 1:
                                    self._enqueue_action({
                                        'action': 'OpenObject',
                                        'objectId': act['objectId'],
                                        'agent_id': act['agent_id']
                                    }, front=True)
                                    new_act = dict(act)
                                    new_act["retry"] = retry + 1
                                    self._enqueue_action(new_act, front=True)
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'ToggleObjectOn':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="ToggleObjectOn",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[ToggleObjectOn] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'ToggleObjectOff':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="ToggleObjectOff",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[ToggleObjectOff] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'OpenObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="OpenObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[OpenObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'CloseObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="CloseObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[CloseObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'SliceObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="SliceObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[SliceObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'ThrowObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="ThrowObject",
                            moveMagnitude=7,
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[ThrowObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'BreakObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="BreakObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(f"[BreakObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'Done':
                        multi_agent_event = c.step(action="Done")

                except Exception as e:
                    print(f"[ExecActions] Exception: {e}")

                if img_counter % 50 == 0:
                    print(f"[ExecActions] processed={img_counter}, queue={self._queue_total_len()}")

                # Checker update (SmartLLM-style)
                if self.checker is not None and multi_agent_event is not None:
                    try:
                        agent_id = act.get("agent_id", 0)
                        success = multi_agent_event.events[agent_id].metadata.get("lastActionSuccess", True)
                        action_name = act.get("action")
                        obj_id = act.get("objectId")
                        if action_name in [
                            "PickupObject",
                            "PutObject",
                            "ToggleObjectOn",
                            "ToggleObjectOff",
                            "OpenObject",
                            "CloseObject",
                            "SliceObject",
                            "BreakObject",
                            "ThrowObject",
                        ]:
                            readable_obj = self._convert_object_id_to_readable(obj_id)
                            action_str = f"{action_name}({readable_obj})"
                            self._update_inventory(agent_id)
                            self.checker.perform_metric_check(action_str, success, self.inventory[agent_id])
                    except Exception:
                        pass

                # 화면 뷰
                if multi_agent_event is not None:
                    try:
                        for i, e in enumerate(multi_agent_event.events):
                            cv2.imshow(f'Robot {i+1}', e.cv2img)
                        # 탑뷰
                        if c.last_event.events[0].third_party_camera_frames:
                            top_view_rgb = cv2.cvtColor(
                                c.last_event.events[0].third_party_camera_frames[-1],
                                cv2.COLOR_BGR2RGB
                            )
                            cv2.imshow('Top View', top_view_rgb)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        pass

                img_counter += 1
                # Notify waiting threads that an action for this agent was processed
                try:
                    agent_id = act.get("agent_id", None)
                    if agent_id is not None:
                        with self.action_cv:
                            if agent_id < len(self.agent_action_counters):
                                self.agent_action_counters[agent_id] += 1
                            self.action_cv.notify_all()
                except Exception:
                    pass
            else:
                time.sleep(0.05)

    # -----------------------------
    # High-level Actions(로봇 동작 제어)
    # -----------------------------
    def _find_object_id(self, obj_pattern: str) -> Optional[str]:
        """객체 이름(예: "Apple")을 기반으로 시뮬레이션 내의 고유 ID 찾는 함수"""
        objs = [obj["objectId"] for obj in self.controller.last_event.metadata["objects"]]
        for obj in objs:
            if re.match(obj_pattern, obj, re.IGNORECASE):
                return obj
        return None

    def _cache_key(self, agent_id: int, pattern: str) -> Tuple[int, str]:
        return (agent_id, pattern.strip().lower())

    def _get_cached_receptacle(self, agent_id: int, pattern: str) -> Optional[str]:
        return self.receptacle_cache.get(self._cache_key(agent_id, pattern))

    def _set_cached_receptacle(self, agent_id: int, pattern: str, obj_id: Optional[str]) -> None:
        if obj_id:
            self.receptacle_cache[self._cache_key(agent_id, pattern)] = obj_id

    def _find_object_with_center(self, obj_pattern: str) -> Tuple[Optional[str], Optional[dict]]:
        """객체 이름(예: "Apple")을 기반으로 시뮬레이션 내의 좌표를 찾는 함수"""
        objs = self.controller.last_event.metadata["objects"]
        for obj in objs:
            if re.match(obj_pattern, obj["objectId"], re.IGNORECASE):
                center = obj.get("axisAlignedBoundingBox", {}).get("center")
                if center and center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    return obj["objectId"], center
        return None, None

    def _find_closest_receptacle(self, recp_pattern: str, agent_id: int) -> Optional[str]:
        """해당 agent 기준으로 가장 가까운 receptacle 찾기"""
        agent_meta = self.controller.last_event.events[agent_id].metadata
        agent_pos = agent_meta["agent"]["position"]

        best_id = None
        best_dist = float('inf')

        for obj in self.controller.last_event.metadata["objects"]:
            if re.match(recp_pattern, obj["objectId"], re.IGNORECASE):
                obj_pos = obj.get("position", {})
                dist = ((agent_pos["x"] - obj_pos.get("x", 0))**2 +
                        (agent_pos["z"] - obj_pos.get("z", 0))**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj["objectId"]

        return best_id

    def _agent_has_pending_actions(self, agent_id: int) -> bool:
        """해당 로봇의 액션이 action_queue에 남아 있는지 확인"""
        with self.action_lock:
            if not self.action_queues:
                return False
            if agent_id < 0 or agent_id >= len(self.action_queues):
                return False
            return len(self.action_queues[agent_id]) > 0

    def _identify_blocking_robot(self, agent_id: int, threshold: float = 0.65) -> Optional[int]:
        try:
            my_pos = self.controller.last_event.events[agent_id].metadata["agent"]["position"]

            best = None
            best_dist = float("inf")

            for other_id in range(self.no_robot):
                if other_id == agent_id:
                    continue

                # 상대가 이미 움직이는 중이면(=그 로봇 액션이 큐에 있으면) 방해로 안 봄
                if self._agent_has_pending_actions(other_id):
                    continue

                other_pos = self.controller.last_event.events[other_id].metadata["agent"]["position"]
                dist = ((my_pos["x"] - other_pos["x"])**2 + (my_pos["z"] - other_pos["z"])**2) ** 0.5

                if dist < threshold and dist < best_dist:
                    best_dist = dist
                    best = other_id

            return best
        except Exception:
            return None
        
    def _issue_yield_request(self, blocking_id: int, requester_id: int, target_object: str) -> bool:
        now = time.time()
        with self.bb_lock:
            # 너무 자주 갱신하는 거 방지(예: 1초 쿨다운)
            old = self.yield_requests.get(blocking_id)
            if old and (now - old.timestamp) < self.yield_cooldown_s:
                return False
            last_pair = self.last_yield_request.get((blocking_id, requester_id))
            if last_pair and (now - last_pair) < self.yield_cooldown_s:
                return False

            # 상호 데드락 방지: requester도 blocking_id에게 양보 요청을 받고 있으면
            # agent_id가 높은 쪽이 양보 (낮은 쪽이 우선권)
            mutual = self.yield_requests.get(requester_id)
            if mutual and mutual.requester_id == blocking_id:
                if requester_id > blocking_id:
                    # 내가 우선순위 낮음 → 내가 양보해야 함, 상대에게 양보 요청 안 함
                    return False
                else:
                    # 내가 우선순위 높음 → 상대의 양보 요청 제거하고 내 요청 등록
                    del self.yield_requests[requester_id]

            self.yield_requests[blocking_id] = YieldRequest(
                requester_id=requester_id,
                timestamp=now,
                reason=f"yield_for_{target_object}",
                target_object=target_object,
                attempts=0,
                last_distance=0.0,
                next_time=now,
            )
            self.last_yield_request[(blocking_id, requester_id)] = now
        return True
    def _find_yield_position(self, blocker_pos: dict, requester_pos: dict, step: float = 0.75) -> Optional[dict]:
        # blocker -> requester 벡터
        vx = requester_pos["x"] - blocker_pos["x"]
        vz = requester_pos["z"] - blocker_pos["z"]
        norm = (vx*vx + vz*vz) ** 0.5
        if norm < 1e-6:
            return None

        vx /= norm
        vz /= norm

        # 수직 방향 2개 후보
        candidates = [
            ( -vz,  vx),  # left
            (  vz, -vx),  # right
        ]

        # reachable 중에서 가장 가까운 점을 고르기
        best = None
        best_d = float("inf")

        for px, pz in candidates:
            target_x = blocker_pos["x"] + px * step
            target_z = blocker_pos["z"] + pz * step

            # reachable_positions_는 dict list: {"x","y","z"}
            for rp in self.reachable_positions_:
                dx = rp["x"] - target_x
                dz = rp["z"] - target_z
                d = (dx*dx + dz*dz) ** 0.5
                if d < best_d:
                    best_d = d
                    best = rp

        # 너무 먼 점이면 실패 처리(선택)
        if best is None or best_d > 1.0:
            return None
        return dict(x=best["x"], y=best["y"], z=best["z"])

    def _monitor_path_clear_requests(self):
        while not self.task_over:
            req_items = []
            with self.bb_lock:
                # 복사해서 락 오래 안 잡기
                req_items = list(self.yield_requests.items())

            for blocking_id, req in req_items:
                try:
                    # 이미 blocker가 뭔가 하느라 바쁘면(=pending) 이번 턴은 스킵
                    if self._agent_has_pending_actions(blocking_id):
                        continue

                    blocker_pos = self.controller.last_event.events[blocking_id].metadata["agent"]["position"]
                    requester_pos = self.controller.last_event.events[req.requester_id].metadata["agent"]["position"]
                    dist = ((blocker_pos["x"] - requester_pos["x"])**2 + (blocker_pos["z"] - requester_pos["z"])**2) ** 0.5

                    # 이미 충분히 멀어졌다면 요청 종료
                    if dist >= (self.yield_clear_distance + self.yield_margin):
                        with self.bb_lock:
                            if self.yield_requests.get(blocking_id) == req:
                                del self.yield_requests[blocking_id]
                        continue

                    # 요청 쿨다운
                    now = time.time()
                    if req.next_time and now < req.next_time:
                        continue

                    if req.last_distance <= 0:
                        req.last_distance = dist

                    # 단계적으로 더 큰 step 시도
                    step_candidates = [0.75, 1.25, 1.75]
                    step = step_candidates[min(req.attempts, len(step_candidates) - 1)]
                    target_position = self._find_yield_position(blocker_pos, requester_pos, step=step)
                    if not target_position:
                        req.attempts += 1
                        req.next_time = now + self.yield_retry_delay_s
                        continue

                    # blocking 로봇에게 1스텝만 이동 명령 (결과 기반 평가)
                    self._enqueue_action({
                        "action": "ObjectNavExpertAction",
                        "position": target_position,
                        "agent_id": blocking_id
                    }, front=False)
                    req.attempts += 1
                    req.next_time = now + self.yield_retry_delay_s

                    # 거리 증가 체크는 다음 루프에서 수행
                    req.last_distance = dist

                except Exception:
                    pass

            time.sleep(0.1)


    def _check_other_robot_blocking(self, agent_id: int, threshold: float = 1.0) -> bool:
        """충돌 방지 로직, 다른 로봇이 길을 막고 있는지 확인하고, 막혀 있다면 옆으로 비켜가는 회피 기동해주는 함수"""
        try:
            my_meta = self.controller.last_event.events[agent_id].metadata
            my_pos = my_meta["agent"]["position"]

            for other_id in range(self.no_robot):
                if other_id == agent_id:
                    continue
                other_meta = self.controller.last_event.events[other_id].metadata
                other_pos = other_meta["agent"]["position"]

                dist = ((my_pos["x"] - other_pos["x"])**2 +
                        (my_pos["z"] - other_pos["z"])**2)**0.5
                if dist < threshold:
                    return True
        except:
            pass
        return False

    def _try_avoid_collision(self, agent_id: int):
        """충돌 방지 로직, 다른 로봇이 길을 막고 있는지 확인하고, 막혀 있다면 옆으로 비켜가는 회피 기동해주는 함수"""

        avoidance_actions = [ #피하는 동작
            ('RotateRight', 45),
            ('MoveAhead', None),
            ('RotateLeft', 90),
            ('MoveAhead', None),
            ('RotateRight', 45),
        ]

        for action, param in avoidance_actions:
            if action == 'RotateRight':
                self._enqueue_action({
                    'action': 'RotateRight',
                    'degrees': param,
                    'agent_id': agent_id
                })
            elif action == 'RotateLeft':
                self._enqueue_action({
                    'action': 'RotateLeft',
                    'degrees': param,
                    'agent_id': agent_id
                })
            elif action == 'MoveAhead':
                self._enqueue_action({
                    'action': 'MoveAhead',
                    'agent_id': agent_id
                })
            time.sleep(0.3)

    def GoToObject(self, agent_id: int, dest_obj: str) -> bool:
        """로봇을 목적지까지 이동시켜주는 함수, ObjectNavExpertAction을 활용해 장애물을 피해가며 목적지 도달 후 물체를 바라보도록 정렬시킴 """
        print(f"[Robot{agent_id+1}] Going to {dest_obj}")

        # 대상 물체 찾기 및 위치 파악
        dest_obj_id, dest_obj_center = self._find_object_with_center(dest_obj)
        if not dest_obj_id or not dest_obj_center:
            print(f"[Robot{agent_id+1}] Cannot find {dest_obj}")
            return False

        print(f"[Robot{agent_id+1}] Target: {dest_obj_id} at {dest_obj_center}")

        dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

        # 내비게이션(이동) 파라미터 설정
        dist_goal = 10.0
        prev_dist_goal = 10.0
        count_since_update = 0 # 이동 정체 횟수 카운트
        clost_node_location = [0] # 도달 가능한 지점 인덱스
        goal_thresh = 0.25 # 목표 지점 도달 인정 거리
        max_iterations = 100   # 최대 반복 횟수 (무한 루프 방지)
        max_recoveries = 3
        recovery_attempts = 0
        collision_retry_count = 0 # 충돌 회피 시도 횟수
        max_collision_retries = 3 # 최대 충돌 회피 시도 횟수
        prev_rot_only = self.nav_rotation_only_count[agent_id] if agent_id < len(self.nav_rotation_only_count) else 0

        # Oscillation 감지용 위치 히스토리 (실제 이동한 위치만 기록)
        position_history: List[Tuple[float, float]] = []  # (x, z) 리스트
        oscillation_count = 0
        oscillation_cooldown = 0  # recovery 후 쿨다운 카운터

        # 대상 물체와 가장 가까운 '이동 가능한 지점' 가져오기
        crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)

        iteration = 0
        while dist_goal > goal_thresh and iteration < max_iterations:
            iteration += 1

            # 현재 로봇의 위치 정보(메타데이터) 가져오기
            metadata = self.controller.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
            }

            prev_dist_goal = dist_goal
            # 목표 지점과 현재 로봇 위치 사이의 거리 계산
            dist_goal = distance_pts(
                [location['x'], location['y'], location['z']],
                crp[0]
            )

            dist_del = abs(dist_goal - prev_dist_goal)

            # 진행 상황 로그 (10회마다)
            if iteration % 10 == 0:
                print(f"[Robot{agent_id+1}] Nav {iteration}/{max_iterations}: dist={dist_goal:.2f}, stall={count_since_update}, osc={oscillation_count}, queue={self._queue_total_len()}")

            # --- Oscillation (앞뒤 반복) 감지 ---
            cur_xz = (location['x'], location['z'])

            # 쿨다운 중이면 감지 건너뜀
            if oscillation_cooldown > 0:
                oscillation_cooldown -= 1
            else:
                # 실제로 이동한 경우에만 히스토리에 기록 (회전만 한 경우 제외)
                moved = True
                if position_history:
                    last_xz = position_history[-1]
                    move_dist = ((cur_xz[0] - last_xz[0])**2 + (cur_xz[1] - last_xz[1])**2) ** 0.5
                    if move_dist < self.nav_oscillation_move_thresh:
                        moved = False

                if moved:
                    # 히스토리에서 3개 이전 위치 중 재방문 확인 (실제 이동 기반)
                    revisit = False
                    if len(position_history) >= 3:
                        for old_xz in position_history[:-2]:
                            dx = cur_xz[0] - old_xz[0]
                            dz = cur_xz[1] - old_xz[1]
                            if (dx*dx + dz*dz) ** 0.5 < self.nav_oscillation_radius:
                                revisit = True
                                break
                    if revisit:
                        oscillation_count += 1
                    else:
                        oscillation_count = max(0, oscillation_count - 1)

                    position_history.append(cur_xz)
                    if len(position_history) > self.nav_position_history_size:
                        position_history.pop(0)

            # Oscillation 감지 시 강제 경로 전환
            if oscillation_count >= self.nav_oscillation_threshold:
                print(f"[Robot{agent_id+1}] Oscillation detected! Switching to alternative path")
                # 뒤로 한 칸 이동 후 다른 접근 지점으로 전환
                self._enqueue_and_wait({
                    'action': 'MoveBack',
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=5.0)
                self._enqueue_and_wait({
                    'action': 'RotateRight',
                    'degrees': 60 + random.randint(0, 60),
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=5.0)
                self._enqueue_and_wait({
                    'action': 'MoveAhead',
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=5.0)

                clost_node_location[0] += 1
                max_positions = len(self.reachable_positions) // 5
                if clost_node_location[0] >= max_positions:
                    clost_node_location[0] = 0
                crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)
                oscillation_count = 0
                count_since_update = 0
                position_history.clear()
                oscillation_cooldown = self.nav_oscillation_cooldown_iters
                time.sleep(0.1)
                continue

            # 로봇이 갇혔거나 멈췄는지 확인
            if dist_del < 0.15:
                count_since_update += 1
            else:
                count_since_update = 0
                collision_retry_count = 0  # 이동 중이면 충돌 카운트 초기화

            # rotation-only 반복 감지 시 강제 재탐색
            if agent_id < len(self.nav_rotation_only_count):
                rot_only = self.nav_rotation_only_count[agent_id]
                if rot_only - prev_rot_only >= self.nav_rotation_only_threshold:
                    # 강제 회피: 뒤로 한 칸 + 약간 회전
                    self._enqueue_action({
                        'action': 'MoveBack',
                        'agent_id': agent_id
                    }, front=True)
                    self._enqueue_action({
                        'action': 'RotateRight',
                        'degrees': 45,
                        'agent_id': agent_id
                    }, front=True)
                    clost_node_location[0] += 1
                    max_positions = len(self.reachable_positions) // 5
                    if clost_node_location[0] >= max_positions:
                        clost_node_location[0] = 0
                    crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)
                    prev_rot_only = rot_only
                    time.sleep(0.1)

            # 로봇끼리 서로 길을 막고 있는지 확인
            if count_since_update >= 5:
                blocking = self._identify_blocking_robot(agent_id, threshold=0.65)
                if blocking is not None and blocking != agent_id:
                    issued = self._issue_yield_request(
                        blocking_id=blocking,
                        requester_id=agent_id,
                        target_object=dest_obj
                    )
                    if issued:
                        print(f"[Robot{agent_id+1}] 요청: Robot{blocking+1} 길 비켜줘 ({dest_obj})")
                    count_since_update = 0
                    collision_retry_count += 1
                    time.sleep(0.1)
                    continue
                else:
                    # 다른 로봇이 막고 있지 않은데도 stuck → 환경 장애물에 갇힘
                    collision_retry_count += 1

                    if collision_retry_count >= max_collision_retries:
                        # 텔레포트 대신 우회/재탐색으로 처리
                        clost_node_location[0] += 1
                        count_since_update = 0
                        collision_retry_count = 0
                        max_positions = len(self.reachable_positions) // 5
                        if clost_node_location[0] >= max_positions:
                            print(f"[Robot{agent_id+1}] Exhausted reachable positions, stopping navigation")
                            break
                        crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)
                        time.sleep(0.1)
                    else:
                        count_since_update = 0
                    continue

            # 정상 이동 가능 시 경로 최적화 액션 수행
            if count_since_update < 5:
                ok = self._enqueue_and_wait({
                    'action': 'ObjectNavExpertAction',
                    'position': dict(x=crp[0][0], y=crp[0][1], z=crp[0][2]),
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=10.0)
                if not ok:
                    # if action not processed in time, continue loop to avoid stale state
                    time.sleep(0.1)
            else:
                # 5회 이상 정체 시, 다음으로 가까운 이동 가능 지점으로 목표 업데이트
                clost_node_location[0] += 1
                count_since_update = 0

                # 인덱스 범위 초과 확인 (안전 장치)
                max_positions = len(self.reachable_positions) // 5
                if clost_node_location[0] >= max_positions:
                    print(f"[Robot{agent_id+1}] Exhausted reachable positions, stopping navigation")
                    break

                crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)

            time.sleep(0.1)

        if iteration >= max_iterations and dist_goal > goal_thresh:
            # Recovery loop: re-sample approach + wait + retry (limited)
            while recovery_attempts < max_recoveries and dist_goal > goal_thresh:
                recovery_attempts += 1
                print(f"[Robot{agent_id+1}] Navigation timeout: recovery {recovery_attempts}/{max_recoveries}")

                # 1) wait briefly to let others move
                time.sleep(0.6)

                # 2) re-sample a different approach point
                clost_node_location[0] += 1
                max_positions = len(self.reachable_positions) // 5
                if clost_node_location[0] >= max_positions:
                    clost_node_location[0] = 0
                crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)

                # 3) retry a short navigation window (15회로 축소, stall 감지 포함)
                recovery_stall = 0
                prev_recovery_dist = dist_goal
                for ri in range(15):
                    metadata = self.controller.last_event.events[agent_id].metadata
                    location = {
                        "x": metadata["agent"]["position"]["x"],
                        "y": metadata["agent"]["position"]["y"],
                        "z": metadata["agent"]["position"]["z"],
                        "rotation": metadata["agent"]["rotation"]["y"],
                    }
                    dist_goal = distance_pts(
                        [location['x'], location['y'], location['z']],
                        crp[0]
                    )
                    if dist_goal <= goal_thresh:
                        break
                    # recovery 중에도 진행 상황 로그
                    if ri % 5 == 0:
                        print(f"[Robot{agent_id+1}] Recovery nav {ri}/15: dist={dist_goal:.2f}")
                    # stall 감지: recovery 중에도 진전 없으면 빠르게 포기
                    if abs(dist_goal - prev_recovery_dist) < 0.1:
                        recovery_stall += 1
                    else:
                        recovery_stall = 0
                    prev_recovery_dist = dist_goal
                    if recovery_stall >= 5:
                        print(f"[Robot{agent_id+1}] Recovery stalled, trying next approach")
                        break
                    self._enqueue_and_wait({
                        'action': 'ObjectNavExpertAction',
                        'position': dict(x=crp[0][0], y=crp[0][1], z=crp[0][2]),
                        'agent_id': agent_id
                    }, agent_id=agent_id, timeout=5.0)

            if dist_goal > goal_thresh:
                print(f"[Robot{agent_id+1}] Navigation timeout, giving up")
                return False

        # [회전 정렬] 로봇이 목적지 물체를 정면으로 바라보도록 회전
        try:
            metadata = self.controller.last_event.events[agent_id].metadata
            robot_location = {
                "x": metadata["agent"]["position"]["x"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
            }

            # 로봇에서 물체를 향하는 벡터 계산
            robot_object_vec = [
                dest_obj_pos[0] - robot_location['x'],
                dest_obj_pos[2] - robot_location['z']
            ]

            vec_magnitude = np.linalg.norm(robot_object_vec)
            if vec_magnitude > 0.01:  # 0으로 나누기 방지
                y_axis = [0, 1]
                unit_y = y_axis / np.linalg.norm(y_axis)
                unit_vector = robot_object_vec / vec_magnitude

                # 물체와의 각도 계산 후 회전 방향(좌/우) 및 각도 결정
                angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
                angle = 360 * angle / (2 * np.pi)
                angle = (angle + 360) % 360
                rot_angle = angle - robot_location['rotation']

                if rot_angle > 0:
                    self._enqueue_action({
                        'action': 'RotateRight',
                        'degrees': abs(rot_angle),
                        'agent_id': agent_id
                    })
                else:
                    self._enqueue_action({
                        'action': 'RotateLeft',
                        'degrees': abs(rot_angle),
                        'agent_id': agent_id
                    })
        except Exception as e:
            print(f"[Robot{agent_id+1}] Alignment error: {e}")

        if dist_goal > goal_thresh:
            print(f"[Robot{agent_id+1}] FAIL to reach {dest_obj}, aborting")
            return False

        print(f"[Robot{agent_id+1}] Reached {dest_obj}")
        time.sleep(0.1)
        return True

    def PickupObject(self, agent_id: int, obj_pattern: str):
        """Pick up object."""
        print(f"[Robot{agent_id+1}] Picking up {obj_pattern}")

        obj_id, _ = self._find_object_with_center(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        print(f"[Robot{agent_id+1}] Picking up {obj_id}")
        self._enqueue_action({
            'action': 'PickupObject',
            'objectId': obj_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def PutObject(self, agent_id: int, obj_pattern: str, recp_pattern: str):
        """Put held object on/in receptacle."""
        print(f"[Robot{agent_id+1}] Putting {obj_pattern} on/in {recp_pattern}")

        # Find closest receptacle
        recp_id = self._get_cached_receptacle(agent_id, recp_pattern)
        if not recp_id:
            recp_id = self._find_closest_receptacle(recp_pattern, agent_id)
            self._set_cached_receptacle(agent_id, recp_pattern, recp_id)
        if not recp_id:
            print(f"[Robot{agent_id+1}] Cannot find receptacle {recp_pattern}")
            return

        print(f"[Robot{agent_id+1}] Putting on {recp_id}")
        # NOTE: PutObject uses objectId for receptacle!
        self._enqueue_action({
            'action': 'PutObject',
            'objectId': recp_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def OpenObject(self, agent_id: int, obj_pattern: str):
        """Open object."""
        print(f"[Robot{agent_id+1}] Opening {obj_pattern}")

        obj_id = self._get_cached_receptacle(agent_id, obj_pattern)
        if not obj_id:
            obj_id = self._find_object_id(obj_pattern)
            self._set_cached_receptacle(agent_id, obj_pattern, obj_id)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return
        time.sleep(1)

        self._enqueue_action({
            'action': 'OpenObject',
            'objectId': obj_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def CloseObject(self, agent_id: int, obj_pattern: str):
        """Close object."""
        print(f"[Robot{agent_id+1}] Closing {obj_pattern}")

        obj_id = self._get_cached_receptacle(agent_id, obj_pattern)
        if not obj_id:
            obj_id = self._find_object_id(obj_pattern)
            self._set_cached_receptacle(agent_id, obj_pattern, obj_id)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return
        time.sleep(1)

        self._enqueue_action({
            'action': 'CloseObject',
            'objectId': obj_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def SwitchOn(self, agent_id: int, obj_pattern: str):
        """Turn on object."""
        print(f"[Robot{agent_id+1}] Switching on {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return
        time.sleep(1)

        self._enqueue_action({
            'action': 'ToggleObjectOn',
            'objectId': obj_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def SwitchOff(self, agent_id: int, obj_pattern: str):
        """Turn off object."""
        print(f"[Robot{agent_id+1}] Switching off {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return
        time.sleep(1)

        self._enqueue_action({
            'action': 'ToggleObjectOff',
            'objectId': obj_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def SliceObject(self, agent_id: int, obj_pattern: str):
        """Slice object."""
        print(f"[Robot{agent_id+1}] Slicing {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return
        time.sleep(1)

        self._enqueue_action({
            'action': 'SliceObject',
            'objectId': obj_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def BreakObject(self, agent_id: int, obj_pattern: str):
        """Break object."""
        print(f"[Robot{agent_id+1}] Breaking {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return
        time.sleep(1)

        self._enqueue_action({
            'action': 'BreakObject',
            'objectId': obj_id,
            'agent_id': agent_id
        })
        time.sleep(1)

    def ThrowObject(self, agent_id: int, obj_pattern: str):
        """Throw held object."""
        print(f"[Robot{agent_id+1}] Throwing {obj_pattern}")

        self._enqueue_action({
            'action': 'ThrowObject',
            'objectId': obj_pattern,  # Not actually used, but kept for consistency
            'agent_id': agent_id
        })
        time.sleep(1)

    # -----------------------------
    # PDDL Action Parser & Executor 실행기
    # -----------------------------
    def _parse_action(self, action_str: str) -> Tuple[str, str, List[str]]:
        """PDDL 형식의 문자열(예: pickupobject robot1 apple)을 분석하여 명령 종류, 로봇, 대상 물체로 쪼개는 함수"""
        action_str = re.sub(r"\s*\(\d+\)\s*$", "", action_str).strip()
        parts = action_str.split()
        if len(parts) < 1:
            return ("", "", [])
        if len(parts) == 1:
            return (parts[0].lower(), "", [])
        action_type = parts[0].lower()
        robot = parts[1]
        objects = parts[2:] if len(parts) > 2 else []
        return (action_type, robot, objects)

    def _execute_pddl_action(self, agent_id: int, action_str: str):
        """파싱된 결과에 따라 위에서 정의한 GoToObject, PickupObject, OpenObject 등의 고수준 함수를 호출하여 실제 명령 큐에 넣는 함수"""
        atype, _, objs = self._parse_action(action_str)

        if atype == "gotoobject" and len(objs) >= 1:
            self.GoToObject(agent_id, objs[0])

        elif atype == "pickupobject" and len(objs) >= 1:
            if not self.GoToObject(agent_id, objs[0]):
                return
            self.PickupObject(agent_id, objs[0])

        elif atype == "putobject" and len(objs) >= 2:
            if not self.GoToObject(agent_id, objs[1]):
                return
            self.PutObject(agent_id, objs[0], objs[1])

        elif atype == "putobjectinfridge" and len(objs) >= 1:
            if not self.GoToObject(agent_id, "Fridge"):
                return
            self.OpenObject(agent_id, "Fridge")
            self.PutObject(agent_id, objs[0], "Fridge")
            self.CloseObject(agent_id, "Fridge")

        elif atype == "openobject" and len(objs) >= 1:
            self.OpenObject(agent_id, objs[0])

        elif atype == "closeobject" and len(objs) >= 1:
            self.CloseObject(agent_id, objs[0])

        elif atype == "openfridge":
            self.OpenObject(agent_id, "Fridge")

        elif atype == "closefridge":
            self.CloseObject(agent_id, "Fridge")

        elif atype == "switchon" and len(objs) >= 1:
            self.SwitchOn(agent_id, objs[0])

        elif atype == "switchoff" and len(objs) >= 1:
            self.SwitchOff(agent_id, objs[0])

        elif atype == "sliceobject" and len(objs) >= 1:
            self.SliceObject(agent_id, objs[0])

        elif atype == "breakobject" and len(objs) >= 1:
            self.BreakObject(agent_id, objs[0])

        elif atype == "throwobject" and len(objs) >= 1:
            if len(objs) >= 2:
                # target이 있으면 GoTo(target) + PutObject로 변환 (던지지 않고 놓기)
                target = objs[1]
                print(f"[Robot{agent_id+1}] throwobject → GoTo({target}) + PutObject 변환")
                if not self.GoToObject(agent_id, target):
                    return
                self.PutObject(agent_id, objs[0], target)
            else:
                # target이 없으면 기존 throw 사용
                self.ThrowObject(agent_id, objs[0])

        elif atype == "drophandobject":
            if len(objs) >= 2:
                # target(receptacle)이 있으면 PutObject 사용 (놓기)
                target = objs[1]
                print(f"[Robot{agent_id+1}] drophandobject → PutObject({objs[0]} on {target})")
                self.PutObject(agent_id, objs[0], target)
            elif len(objs) >= 1:
                # target 없으면 그냥 drop
                print(f"[Robot{agent_id+1}] drophandobject → PutObject({objs[0]})")
                self.PutObject(agent_id, objs[0], objs[0])
            else:
                # 아무 정보도 없으면 AI2Thor DropHandObject 사용
                self._enqueue_action({
                    'action': 'PutObject',
                    'objectId': '',
                    'agent_id': agent_id
                })
                time.sleep(1)

        elif atype == "pushobject" and len(objs) >= 1:
            if not self.GoToObject(agent_id, objs[0]):
                return
            obj_id = self._find_object_id(objs[0])
            if obj_id:
                self._enqueue_action({
                    'action': 'PushObject',
                    'objectId': obj_id,
                    'agent_id': agent_id
                })
            time.sleep(1)

        elif atype == "pullobject" and len(objs) >= 1:
            if not self.GoToObject(agent_id, objs[0]):
                return
            obj_id = self._find_object_id(objs[0])
            if obj_id:
                self._enqueue_action({
                    'action': 'PullObject',
                    'objectId': obj_id,
                    'agent_id': agent_id
                })
            time.sleep(1)

        else:
            print(f"[Robot{agent_id+1}] SKIP/UNKNOWN: {action_str}")

    # -----------------------------
    # AI2-THOR 관리 함수 모음
    # -----------------------------

    @staticmethod
    def spawn_and_get_positions(
        floor_plan: int, agent_count: int
    ) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
        """
        AI2-THOR를 잠깐 시작하여 로봇 스폰 좌표 + 오브젝트 좌표를 가져오고 종료.

        Returns:
            robot_positions: {robot_id(1-based): (x, y, z)}
            object_positions: {object_name_lower: (x, y, z)}
        """
        controller = Controller(height=300, width=300)
        controller.reset(f"FloorPlan{floor_plan}")
        controller.step(dict(
            action='Initialize', agentMode="default", snapGrid=False,
            gridSize=0.25, rotateStepDegrees=20,
            visibilityDistance=100, fieldOfView=90, agentCount=agent_count
        ))

        # 로봇 랜덤 배치 (start_ai2thor과 동일 로직)
        reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        used: List[dict] = []
        min_distance = 1.5

        for i in range(agent_count):
            best_pos = None
            best_min_dist = -1
            for _ in range(50):
                candidate = random.choice(reachable)
                if not used:
                    best_pos = candidate
                    break
                min_d = min(
                    ((candidate["x"] - p["x"])**2 + (candidate["z"] - p["z"])**2)**0.5
                    for p in used
                )
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = candidate
                if min_d >= min_distance:
                    break
            if best_pos:
                controller.step(dict(action="Teleport", position=best_pos, agentId=i))
                used.append(best_pos)

        # 로봇 좌표 수집
        robot_positions: Dict[int, Tuple[float, float, float]] = {}
        for i in range(agent_count):
            pos = controller.last_event.events[i].metadata["agent"]["position"]
            robot_positions[i + 1] = (pos["x"], pos["y"], pos["z"])  # 1-based

        # 오브젝트 좌표 수집
        object_positions: Dict[str, Tuple[float, float, float]] = {}
        for obj in controller.last_event.metadata["objects"]:
            name = obj["objectType"].strip().lower()
            p = obj.get("position", {})
            object_positions[name] = (p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0))

        controller.stop()
        print(f"[spawn_and_get_positions] Collected {agent_count} robot positions, {len(object_positions)} object positions")
        return robot_positions, object_positions

    def start_ai2thor(self, floor_plan: int, agent_count: int,
                      spawn_positions: Optional[Dict[int, Tuple[float, float, float]]] = None):
        """시뮬레이터를 초기화하고 로봇들을 생성. spawn_positions가 있으면 해당 좌표에 배치."""
        scene = f"FloorPlan{floor_plan}"
        self.scene_name = scene
        print(f"[AI2-THOR] Starting scene={scene}, agentCount={agent_count}")

        self.no_robot = agent_count
        self.controller = Controller(height=1000, width=1000)
        self.controller.reset(scene)

        # 로봇 초기 설정
        self.controller.step(dict(
            action='Initialize',
            agentMode="default",
            snapGrid=False,
            gridSize=0.25,
            rotateStepDegrees=20,
            visibilityDistance=100,
            fieldOfView=90,
            agentCount=agent_count
        ))

        # 탑뷰(Top-down) 카메라 추가
        event = self.controller.step(action="GetMapViewCameraProperties")
        self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

        # 이동 가능한 위치 정보 가져오기
        self.reachable_positions_ = self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        self.reachable_positions = [
            (p["x"], p["y"], p["z"]) for p in self.reachable_positions_
        ]

        # 로봇 배치
        min_spawn_distance = 1.5  # 로봇 간 최소 스폰 거리
        if spawn_positions:
            # LP에서 결정된 좌표를 가장 가까운 NavMesh 유효 위치로 보정하여 배치
            used_reachable = []  # 이미 사용된 reachable position (x, z) 좌표
            for i in range(agent_count):
                rid = i + 1
                if rid in spawn_positions:
                    pos = spawn_positions[rid]
                    # 가장 가까운 reachable position 찾기 (NavMesh 보장 + 최소 간격)
                    best_rp, best_d = None, float("inf")
                    for rp in self.reachable_positions_:
                        # 다른 로봇과 최소 거리 보장
                        too_close = False
                        for ux, uz in used_reachable:
                            if ((rp["x"] - ux)**2 + (rp["z"] - uz)**2)**0.5 < min_spawn_distance:
                                too_close = True
                                break
                        if too_close:
                            continue
                        d = ((rp["x"] - pos[0])**2 + (rp["z"] - pos[2])**2)**0.5
                        if d < best_d:
                            best_d, best_rp = d, rp
                    if best_rp:
                        used_reachable.append((best_rp["x"], best_rp["z"]))
                        self.controller.step(dict(action="Teleport",
                            position=best_rp, agentId=i, forceAction=True))
                        print(f"[Robot{rid}] Requested ({pos[0]:.2f}, {pos[2]:.2f}) -> snapped to reachable ({best_rp['x']:.2f}, {best_rp['z']:.2f}), dist={best_d:.2f}")
                    else:
                        # fallback: forceAction 텔레포트
                        self.controller.step(dict(action="Teleport",
                            position=dict(x=pos[0], y=pos[1], z=pos[2]),
                            agentId=i, forceAction=True))
                        print(f"[Robot{rid}] WARNING: No reachable position found, force-teleported to ({pos[0]:.2f}, {pos[2]:.2f})")
                    # 실제 위치 확인
                    actual = self.controller.last_event.events[i].metadata["agent"]["position"]
                    print(f"[Robot{rid}] Actual position: ({actual['x']:.2f}, {actual['z']:.2f})")
                else:
                    # 스폰 위치 미지정 로봇 → 다른 로봇과 떨어진 reachable position에 배치
                    best_rp, best_d = None, -1
                    for rp in self.reachable_positions_:
                        rp_tuple = (rp["x"], rp["z"])
                        if rp_tuple in used_reachable:
                            continue
                        min_d = min(
                            (((rp["x"] - ux)**2 + (rp["z"] - uz)**2)**0.5
                             for ux, uz in used_reachable),
                            default=float("inf")
                        )
                        if min_d > best_d:
                            best_d, best_rp = min_d, rp
                    if best_rp:
                        used_reachable.append((best_rp["x"], best_rp["z"]))
                        self.controller.step(dict(action="Teleport",
                            position=best_rp, agentId=i, forceAction=True))
                        print(f"[Robot{rid}] No spawn pos assigned, placed at reachable ({best_rp['x']:.2f}, {best_rp['z']:.2f})")
        else:
            # 기존 랜덤 배치 - 로봇 간 거리를 고려하여 배치
            used_positions = []
            min_distance = 1.5  # 로봇들간의 최소거리

            for i in range(agent_count):
                best_pos = None
                best_min_dist = -1

                # 다른 로봇과 가장 멀리 떨어진 위치를 찾기 위해 최대 50번 시도
                for _ in range(50):
                    candidate = random.choice(self.reachable_positions_)

                    if not used_positions:
                        best_pos = candidate
                        break

                    # 이미 배치된 모든 위치와의 최소 거리 계산
                    min_dist_to_others = min(
                        ((candidate["x"] - p["x"])**2 + (candidate["z"] - p["z"])**2)**0.5
                        for p in used_positions
                    )

                    if min_dist_to_others > best_min_dist:
                        best_min_dist = min_dist_to_others
                        best_pos = candidate

                    if min_dist_to_others >= min_distance:
                        break

                if best_pos:
                    self.controller.step(dict(action="Teleport", position=best_pos, agentId=i))
                    used_positions.append(best_pos)
                    print(f"[Robot{i+1}] Spawned at ({best_pos['x']:.2f}, {best_pos['z']:.2f})")

        # 물체를 더 잘 보기 위해 카메라를 약간 아래로 조절
        for i in range(agent_count):
            self.controller.step(action="LookDown", degrees=35, agentId=i)

        # 사용 가능한 물체 리스트
        objs = [obj["objectType"] for obj in self.controller.last_event.metadata["objects"]]

        # 액션 실행 스레드 시작
        self._build_object_dict()
        self.inventory = ["nothing"] * agent_count
        self.agent_action_counters = [0] * agent_count
        self.action_queues = [deque() for _ in range(agent_count)]
        self.nav_rotation_only_count = [0] * agent_count
        self.rr_index = 0
        self.task_over = False
        self.actions_thread = threading.Thread(target=self._exec_actions)
        self.actions_thread.start()

        self.monitor_thread = threading.Thread(target=self._monitor_path_clear_requests, daemon=True)
        self.monitor_thread.start()

        print("[AI2-THOR] Ready!")

    def stop_ai2thor(self):
        """AI2-THOR 시뮬레이터를 정지하는 함수"""
        self.task_over = True
        if self.actions_thread:
            self.actions_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        try:
            cv2.destroyAllWindows()
        except:
            pass

        try:
            self.controller.stop()
        except:
            pass

        print(f"[AI2-THOR] Stopped. Success: {self.success_exec}/{self.total_exec}")

    # -----------------------------
    # Subtask 실행기
    # -----------------------------
    def _run_subtask(self, plan: SubtaskPlan):
        """하나의 서브태스크에 포함된 모든 액션을 순차적으로 실행하는 함수"""
        agent_id = plan.robot_id - 1  # 1-기반 ID를 0-기반 인덱스로 변환
        print(f"\n[Subtask {plan.subtask_id}] {plan.subtask_name} -> Robot{plan.robot_id}")

        for i, action in enumerate(plan.actions):
            print(f"[Robot{plan.robot_id}] Action {i+1}/{len(plan.actions)}: {action}")
            self._execute_pddl_action(agent_id, action)

        print(f"[Subtask {plan.subtask_id}] Completed!")

    def execute_in_ai2thor(self, floor_plan: int, task_description: Optional[str] = None,
                           agent_count: Optional[int] = None,
                           spawn_positions: Optional[Dict[int, Tuple[float, float, float]]] = None):
        """할당된 모든 서브태스크를 AI2-THOR에서 실행하는 함수.
        agent_count: 소환할 총 로봇 수. None이면 할당된 로봇 ID 최댓값 사용.
        spawn_positions: LP에서 결정된 스폰 좌표. None이면 랜덤 배치.
        """
        if not self.assignment or not self.parallel_groups or not self.subtask_plans:
            raise RuntimeError("Load assignment/DAG/plans first.")

        if agent_count is None:
            agent_count = getattr(self, 'configured_agent_count', None) \
                          or (max(self.assignment.values()) if self.assignment else 1)
        if spawn_positions is None:
            spawn_positions = getattr(self, 'saved_spawn_positions', None)
        self.start_ai2thor(floor_plan=floor_plan, agent_count=agent_count,
                           spawn_positions=spawn_positions)
        if task_description:
            self._init_checker(task_description, self.scene_name)

        # 병렬 그룹별로 서브태스크 그룹화
        groups_to_plans: Dict[int, List[SubtaskPlan]] = defaultdict(list)
        for sid, p in self.subtask_plans.items():
            groups_to_plans[p.parallel_group].append(p)

        print("=" * 60)
        print("[EXEC] Starting Multi-Robot Execution")
        print("=" * 60)

        try:
            for gid in sorted(groups_to_plans.keys()):
                plans = groups_to_plans[gid]
                print(f"\n[Group {gid}] {len(plans)} subtask(s) in parallel")

                # 같은 로봇에 할당된 서브태스크끼리는 순차 실행해야 하므로
                # 로봇별로 묶어서 하나의 스레드 안에서 순차 처리
                robot_plans: Dict[int, List[SubtaskPlan]] = defaultdict(list)
                for p in plans:
                    robot_plans[p.robot_id].append(p)

                def _run_robot_plans(plan_list: List[SubtaskPlan]):
                    for p in plan_list:
                        self._run_subtask(p)

                threads = []
                for _, plan_list in robot_plans.items():
                    # 로봇마다 하나의 스레드: 서로 다른 로봇끼리만 병렬
                    t = threading.Thread(target=_run_robot_plans, args=(plan_list,), daemon=True)
                    threads.append(t)

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                print(f"[Group {gid}] Completed!")

            # 모든 액션 큐가 비워질 때까지 대기
            print("\n[EXEC] Waiting for action queue to empty...")
            while True:
                if self._queue_total_len() == 0:
                    break
                time.sleep(0.5)

            # 종료 전 확인을 위해 잠시 대기
            print("[EXEC] All done! Press 'q' in any window to close.")
            time.sleep(3)

            if self.checker is not None:
                try:
                    coverage = self.checker.get_coverage()
                    transport_rate = self.checker.get_transport_rate()
                    finished = self.checker.check_success()
                    print(f"Coverage: {coverage}, Transport Rate: {transport_rate}, Finished: {finished}")
                except Exception:
                    pass

        finally:
            self.stop_ai2thor()

    # -----------------------------
    # 코드 생성 (chaerin_pddl.py 호환용)
    # -----------------------------
    def run(
        self,
        task_idx: int = 0,
        task_name: str = "task",
        task_description: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """멀티 로봇 실행 코드를 생성하고 파일로 저장"""
        self.load_assignment(task_idx)
        self.load_subtask_dag(task_name)
        self.load_plan_actions()

        code = self._generate_execution_code(task_description=task_description)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(code)
            print(f"[Executor] Saved execution code to: {output_path}")

        return code

    def _generate_execution_code(self, task_description: Optional[str] = None) -> str:
        """멀티 로봇 실행을 위한 파이썬 코드 생성"""
        lines = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated multi-robot execution code.",
            "Run with: python <this_file> --floor-plan <N>",
            '"""',
            "",
            "import argparse",
            "import sys",
            "import os",
            "",
            "import sys",
            "from pathlib import Path",
            "PDL_ROOT = Path(__file__).resolve().parents[2]",
            "sys.path.insert(0, str(PDL_ROOT))",
            'sys.path.insert(0, str(PDL_ROOT / "scripts"))',
            'sys.path.insert(0, str(PDL_ROOT / "resources"))',

            "# Add scripts folder to path",
            "base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
            "scripts_path = os.path.join(base_path, 'scripts')",
            "if scripts_path not in sys.path:",
            "    sys.path.insert(0, scripts_path)",
            "",
            "from MultiRobotExecutor import MultiRobotExecutor, SubtaskPlan",
            "",
            "# --- Robot Assignment ---",
            f"ASSIGNMENT = {self.assignment}  # subtask_id -> robot_id",
            f"PARALLEL_GROUPS = {self.parallel_groups}  # group_id -> [subtask_ids]",
            f"AGENT_COUNT = {getattr(self, 'configured_agent_count', None) or (max(self.assignment.values()) if self.assignment else 1)}",
            f"SPAWN_POSITIONS = {getattr(self, 'saved_spawn_positions', None)}  # LP에서 결정된 스폰 좌표",
            "",
            "# --- Subtask Plans ---",
            "SUBTASK_PLANS = {",
        ]

        for sid, plan in sorted(self.subtask_plans.items()):
            actions_repr = repr(plan.actions)
            lines.append(f"    {sid}: {{")
            lines.append(f"        'name': {repr(plan.subtask_name)},")
            lines.append(f"        'robot_id': {plan.robot_id},")
            lines.append(f"        'actions': {actions_repr},")
            lines.append(f"        'parallel_group': {plan.parallel_group},")
            lines.append(f"    }},")

        lines.append("}")
        lines.append("")
        if task_description is not None:
            lines.append(f"TASK_DESCRIPTION = {repr(task_description)}")
            lines.append("")

        lines.extend([
            "",
            "def main():",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--floor-plan', type=int, default=1)",
            "    args = parser.parse_args()",
            "    ",
            "    executor = MultiRobotExecutor(base_path)",
            "    executor.assignment = ASSIGNMENT",
            "    executor.parallel_groups = PARALLEL_GROUPS",
            "    ",
            "    # Reconstruct subtask_plans",
            "    for sid, data in SUBTASK_PLANS.items():",
            "        executor.subtask_plans[sid] = SubtaskPlan(",
            "            subtask_id=sid,",
            "            subtask_name=data['name'],",
            "            robot_id=data['robot_id'],",
            "            actions=data['actions'],",
            "            parallel_group=data['parallel_group'],",
            "        )",
            "    ",
            "    # Execute in AI2-THOR",
            "    executor.execute_in_ai2thor(",
            "        floor_plan=args.floor_plan,",
            "        task_description=TASK_DESCRIPTION if 'TASK_DESCRIPTION' in globals() else None,",
            "        agent_count=AGENT_COUNT,",
            "        spawn_positions=SPAWN_POSITIONS,",
            "    )",
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ])

        return "\n".join(lines)

    # -----------------------------
    # Pipeline 편의를 위한 함수
    # -----------------------------
    def run_and_execute(
        self,
        task_idx: int = 0,
        task_name: str = "task",
        floor_plan: int = 1,
        agent_count: Optional[int] = None,
        spawn_positions: Optional[Dict[int, Tuple[float, float, float]]] = None,
    ):
        """Load and execute directly."""
        print("\n" + "=" * 60)
        print("MultiRobotExecutor (Load + Execute)")
        print("=" * 60)

        self.load_assignment(task_idx)
        self.load_subtask_dag(task_name)
        self.load_plan_actions()

        self.execute_in_ai2thor(floor_plan=floor_plan, agent_count=agent_count,
                                spawn_positions=spawn_positions)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str,
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--task-name", type=str, default="task")
    parser.add_argument("--floor-plan", type=int, required=True)
    args = parser.parse_args()

    ex = MultiRobotExecutor(args.base_path)
    ex.run_and_execute(
        task_idx=args.task_idx,
        task_name=args.task_name,
        floor_plan=args.floor_plan,
    )


if __name__ == "__main__":
    main()
