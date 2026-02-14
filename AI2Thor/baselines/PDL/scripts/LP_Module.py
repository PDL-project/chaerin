# 작업할당 모듈
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional, Set
from ortools.sat.python import cp_model
import ast
import math

# ----------------------------
# 로봇들의 스킬/능력 추출 함수 모음 부분
# ----------------------------

def get_capacity(robot_info: dict) -> float:
    """
    robots.py가 mass / mass_capacity 섞여있어서 통일해서 읽어오는 함수
    """
    if "mass_capacity" in robot_info:
        return float(robot_info["mass_capacity"])
    if "mass" in robot_info:
        return float(robot_info["mass"])
    return 0.0


def robot_skill_map(robot_ids: List[int], robots_db: List[dict]) -> Dict[int, Set[str]]:
    """
    robots.py에서 각 로봇들의 스킬들을 읽어오는 함수
    (예: 1: {'Pickup', 'Move'})
    """
    m: Dict[int, Set[str]] = {}
    for rid in robot_ids:
        info = robots_db[rid - 1]  # 1-based -> 0-based
        m[rid] = set((s or "").strip() for s in info.get("skills", []))
    return m


def robot_capacity_map(robot_ids: List[int], robots_db: List[dict]) -> Dict[int, float]:
    """
    각 로봇 ID별로 들 수 있는 최대 무게 값을 매핑
    """
    return {rid: get_capacity(robots_db[rid - 1]) for rid in robot_ids}


# ----------------------------
# 환경 object 정보 추출 함수 모음
# ----------------------------

def parse_objects_ai(objects_ai: Any) -> List[dict]:
    """
    LLM이 생성한 물체 정보 데이터를 파이썬의 리스트 객체로 변환

    objects_ai가
    - 이미 list[dict]일 수도 있고
    - 문자열로 "objects = [...]" 형태일 수도 있어서 안전 파싱
    """
    if isinstance(objects_ai, list):
        return objects_ai

    if isinstance(objects_ai, str):
        s = objects_ai.strip()
        # "objects = [...]" 형태면 '=' 뒤만 파싱
        if "=" in s:
            s = s.split("=", 1)[1].strip()
        # python literal list 파싱 (json이 아닌 경우가 많아서)
        try:
            data = ast.literal_eval(s)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


def build_mass_map(objects_ai: Any) -> Dict[str, float]:
    """
    물체 리스트를 순회하며 각 물체의 이름과 무게를 매핑한 딕셔너리를 만드는 함수 (예: {'apple': 0.5})
    """
    objs = parse_objects_ai(objects_ai)
    mass_map: Dict[str, float] = {}
    for o in objs:
        name = str(o.get("name", "")).strip().lower()
        if not name:
            continue
        mass = float(o.get("mass", 0.0) or 0.0)
        mass_map[name] = mass
    return mass_map


# -----------------------------------------
# plan -> 옮겨야하는 오브젝트들의 무게를 계산하는 함수 모음
# -----------------------------------------

def _tok(action_line: str) -> List[str]:
    # 액션 문장(예: pickupobject robot1 apple...)에서 괄호를 제거하고 단어 단위로 쪼개주는 보조 함수
    return action_line.replace("(", " ").replace(")", " ").split()

def picked_objects_from_plan(plan_actions: List[str]) -> Set[str]:
    """
    한 서브태스크의 전체 액션 중 pickup, put, throw 등 물체를 손에 들고 있어야 하는 동작에서 대상 물체들을 추출
    """
    picked: Set[str] = set()
    for a in plan_actions:
        parts = _tok(a)
        if len(parts) < 3:
            continue
        at = parts[0].strip().lower()

        # pickupobject robotX OBJ ...
        if at == "pickupobject" and len(parts) >= 3:
            picked.add(parts[2].strip().lower())

        # put / throw / drop 류도 OBJ를 들고 있었을 가능성이 높으니 포함
        elif at in ("putobject", "putobjectinfridge", "throwobject", "drophandobject") and len(parts) >= 3:
            picked.add(parts[2].strip().lower())

    return picked


def required_mass_from_plan(plan_actions: List[str], mass_map: Dict[str, float]) -> float:
    """
    위에서 추출된 물체들의 무게를 합산하여, 이 작업을 수행할 로봇이 견뎌야 하는 총 무게를 계산하는 함수
    """
    objs = picked_objects_from_plan(plan_actions)
    return sum(float(mass_map.get(o, 0.0)) for o in objs)


# -----------------------------------------
# subtask -> subtask 간의 binding 관계 체크 함수
# -----------------------------------------

def binding_pairs_from_subtask_dag(subtask_dag: Any) -> List[Tuple[int, int]]:
    """
    서브태스크 DAG에서 binding 타입의 엣지를 찾아 "이 작업들은 반드시 같은 로봇이 해야 함"이라는 쌍(Pair) 리스트를 만드는 함수

    SubtaskDAG 구조-> dag.edges에 dependency_type 안에 해당 정보가 들어있음
    type == "binding" 인 엣지는 '같은 로봇 강제'로 처리
    """
    pairs: List[Tuple[int, int]] = []
    if subtask_dag is None:
        return pairs

    for e in getattr(subtask_dag, "edges", []) or []:
        t = (getattr(e, "dependency_type", "") or "").lower().strip()
        if t == "binding":
            pairs.append((int(e.from_id), int(e.to_id)))

    # 중복 제거
    pairs = sorted(list({(a, b) for (a, b) in pairs}))
    return pairs


# -----------------------------------------
# 거리 계산 함수 모음
# -----------------------------------------

def _euclidean_dist(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """3D 유클리드 거리"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def build_position_map(objects_ai: Any) -> Dict[str, Tuple[float, float, float]]:
    """
    objects_ai에서 오브젝트별 좌표 딕셔너리 생성 (예: {'apple': (1.0, 0.9, -2.3)})
    """
    objs = parse_objects_ai(objects_ai)
    pos_map: Dict[str, Tuple[float, float, float]] = {}
    for o in objs:
        name = str(o.get("name", "")).strip().lower()
        pos = o.get("position", {})
        if name and pos:
            pos_map[name] = (float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0)))
    return pos_map


def first_target_object_from_plan(plan_actions: List[str]) -> Optional[str]:
    """
    서브태스크 플랜에서 첫 번째로 이동해야 할 대상 오브젝트를 추출
    gotoobject가 있으면 그 대상, 없으면 첫 번째 액션의 대상 오브젝트 반환
    """
    for a in plan_actions:
        parts = _tok(a)
        if len(parts) >= 3 and parts[0].strip().lower() == "gotoobject":
            return parts[2].strip().lower()
    # gotoobject 없으면 첫 번째 액션의 대상
    for a in plan_actions:
        parts = _tok(a)
        if len(parts) >= 3:
            return parts[2].strip().lower()
    return None


# -----------------------------------------
# Skill 확인 함수
# -----------------------------------------

def build_can_matrix(
    subtasks: List[dict],
    robot_ids: List[int],
    robots_db: List[dict],
    normalize: bool = True
) -> Dict[Tuple[int, int], int]:
    rskills = robot_skill_map(robot_ids, robots_db)
    """
    각 서브태스크가 요구하는 스킬을 로봇이 모두 가졌는지 대조하여, 배정 가능 여부(0 또는 1)를 행렬 형태로 구성해주는 함수
    """
    can: Dict[Tuple[int, int], int] = {}
    for st in subtasks:
        sid = int(st["id"])
        req_raw = st.get("skills", []) or []
        if normalize:
            req = set((s or "").strip() for s in req_raw)
        else:
            req = set(req_raw)

        for rid in robot_ids:
            can[(sid, rid)] = 1 if req.issubset(rskills[rid]) else 0
    return can


# ======================================================
# 메인 프로세스 부분, 할당 실행 함수 (현재 기준: 스킬을 가지고 있는가? mass 능력이 충분한가? binding 포함 관계인가?)
# ======================================================

def assign_subtasks_cp_sat(
    subtasks: List[dict],
    robot_ids: List[int],
    robots_db: List[dict],
    plan_actions_by_subtask: Dict[int, List[str]],
    objects_ai: Any,
    binding_pairs: Optional[List[Tuple[int, int]]] = None,
    cost_by_subtask: Optional[Dict[int, int]] = None,
    time_limit_s: float = 10.0,
    # ↓↓↓ 거리 기반 비용 파라미터
    robot_positions: Optional[Dict[int, Tuple[float, float, float]]] = None,
    object_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    distance_weight: int = 1,
    # ↓↓↓ 로드 밸런싱 파라미터
    balance_mode: str = "sumsq",  # "max" or "sumsq"
    balance_weight: int = 200,
) -> Dict[int, int]:
    """
    CP-SAT 기반 subtask -> robot assignment.

    Returns:
        {subtask_id: robot_id}

    Constraints (hard):
      1) Each subtask assigned to exactly one robot
      2) Skill feasibility (can matrix)
      3) Mass feasibility (required_mass <= capacity)
      4) Binding pairs: same robot must do both subtasks

    Objective (우선순위):
      1순위) 로드 밸런싱 — 로봇 간 작업량 균등 분배 (sumsq, weight=200)
      2순위) 거리 비용 — 로봇→대상 오브젝트 거리, 0.0~1.0 정규화 (weight=1)
    """
    model = cp_model.CpModel()

    # ---- ids ----
    sids = [int(st["id"]) for st in subtasks]

    # ---- can matrix ----
    can = build_can_matrix(subtasks, robot_ids, robots_db)

    # ---- costs ----
    if cost_by_subtask is None:
        cost_by_subtask = {sid: 1 for sid in sids}

    # ---- mass ----
    mass_map = build_mass_map(objects_ai)
    cap = robot_capacity_map(robot_ids, robots_db)

    req_mass: Dict[int, float] = {}
    for sid in sids:
        plan = plan_actions_by_subtask.get(sid, [])
        req_mass[sid] = required_mass_from_plan(plan, mass_map)

    # ---- decision vars: x[sid,rid] ----
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for sid in sids:
        for rid in robot_ids:
            x[(sid, rid)] = model.NewBoolVar(f"x_s{sid}_r{rid}")

    # 1) each subtask exactly one robot
    for sid in sids:
        model.Add(sum(x[(sid, rid)] for rid in robot_ids) == 1)

    # 2) skill feasibility
    for sid in sids:
        for rid in robot_ids:
            if can[(sid, rid)] == 0:
                model.Add(x[(sid, rid)] == 0)

    # 3) mass feasibility
    for sid in sids:
        for rid in robot_ids:
            if req_mass[sid] > cap[rid] + 1e-9:
                model.Add(x[(sid, rid)] == 0)

    # 4) binding (same robot)
    if binding_pairs is None:
        binding_pairs = []
    for (a, b) in binding_pairs:
        a = int(a); b = int(b)
        if a not in sids or b not in sids:
            continue
        for rid in robot_ids:
            model.Add(x[(a, rid)] == x[(b, rid)])

    # ---- distance cost (거리 비용) ----
    # 로봇 위치 → 서브태스크 첫 번째 대상 오브젝트 위치까지의 유클리드 거리를 soft cost로 반영
    # 정규화: 최대 거리 대비 비율로 0.0~1.0 스케일링 (CP-SAT 정수 제약으로 ×1000 → 0~1000)
    dist_cost_map: Dict[Tuple[int, int], int] = {}
    if robot_positions and object_positions:
        raw_dists: Dict[Tuple[int, int], float] = {}
        for sid in sids:
            plan = plan_actions_by_subtask.get(sid, [])
            target_obj = first_target_object_from_plan(plan)
            obj_pos = object_positions.get(target_obj) if target_obj else None
            for rid in robot_ids:
                if obj_pos and rid in robot_positions:
                    raw_dists[(sid, rid)] = _euclidean_dist(robot_positions[rid], obj_pos)
                else:
                    raw_dists[(sid, rid)] = 0.0

        max_dist = max(raw_dists.values()) if raw_dists else 1.0
        if max_dist < 1e-9:
            max_dist = 1.0  # 모든 거리가 0인 경우 방지

        for key, d in raw_dists.items():
            # 0.0~1.0 소수점 정규화 (CP-SAT 정수 제약 → ×1000으로 정밀도 유지)
            dist_cost_map[key] = int((d / max_dist) * 1000)

    # ---- load balancing vars ----
    load_by_robot: Dict[int, cp_model.IntVar] = {}
    for rid in robot_ids:
        load_by_robot[rid] = model.NewIntVar(0, len(sids), f"load_r{rid}")
        model.Add(load_by_robot[rid] == sum(x[(sid, rid)] for sid in sids))

    max_load = None
    if balance_mode == "max":
        max_load = model.NewIntVar(0, len(sids), "max_load")
        model.AddMaxEquality(max_load, [load_by_robot[rid] for rid in robot_ids])

    # ---- objective ----
    # distance cost: 로봇-오브젝트 거리 비용
    distance_cost = sum(dist_cost_map.get((sid, rid), 0) * x[(sid, rid)] for sid in sids for rid in robot_ids)

    # balance cost (1순위)
    if balance_mode == "sumsq":
        # sum of squared loads (convex) -> more even distribution
        sq_terms: List[cp_model.IntVar] = []
        for rid in robot_ids:
            sq = model.NewIntVar(0, len(sids) * len(sids), f"load_sq_r{rid}")
            model.AddMultiplicationEquality(sq, load_by_robot[rid], load_by_robot[rid])
            sq_terms.append(sq)
        balance_cost = sum(sq_terms)
    else:
        # default: minimize maximum load
        balance_cost = max_load if max_load is not None else 0

    # 1순위: 로드 밸런싱 (weight=200), 2순위: 거리 (weight=1)
    model.Minimize(
        balance_weight * balance_cost
        + distance_weight * distance_cost
    )

    # ---- solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            "No feasible assignment. "
            "Check: (1) skill mismatch, (2) mass/capacity too small, (3) binding chain impossible. "
            "If parallel is hard, consider enabling soft parallel."
        )

    # ---- extract assignment ----
    assignment: Dict[int, int] = {}
    for sid in sids:
        for rid in robot_ids:
            if solver.Value(x[(sid, rid)]) == 1:
                assignment[sid] = rid
                break

    return assignment
