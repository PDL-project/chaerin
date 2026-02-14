"""
DAG Module for PDDL Plan Parallelism Analysis
LLM을 사용하여 PDDL plan의 액션, 서브테스크들 간 의존성을 분석하고 DAG를 생성
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import openai



# 1. Action-level DAG 생성 부분

@dataclass
class ActionNode: #그래프의 정점
    id: int
    action: str
    action_type: str
    robot: str
    objects: List[str]
    parallel_group: int = 0


@dataclass
class DAGEdge: #그래프의 엣지(간선) 클래스
    from_id: int
    to_id: int
    dependency_type: str  # "causal": 원인과 결과 (A가 끝나야 B의 조건을 만족함), "resource": 같은 물체나 장소를 사용함 (충돌 방지), "binding": 같은 로봇이 수행해야 함 (물건을 들고 이동하는 경우 등)


@dataclass
class PlanDAG:
    """하나의 서브태스크 내 액션들의 전체 DAG 구조"""
    subtask_name: str
    nodes: List[ActionNode] = field(default_factory=list)
    edges: List[DAGEdge] = field(default_factory=list)
    parallel_groups: Dict[int, List[int]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """JSON 저장을 위한 딕셔너리 변환 (Binding 엣지 별도 추출 포함)"""
        binding_edges = [
            {"from": e.from_id, "to": e.to_id}
            for e in self.edges
            if (e.dependency_type or "").lower() == "binding"
        ]
        return {
            "subtask_name": self.subtask_name,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "parallel_groups": self.parallel_groups,
            "binding_edges": binding_edges
        }


# 2. 서브태스크 레벨(Subtask-level) 데이터 구조

@dataclass
class SubtaskSummary:
    """서브테스크 하나를 요약한 정보(LLM이 만듦)"""
    id: int
    name: str
    objects: List[str] = field(default_factory=list)
    preconds: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)

@dataclass
class SubtaskEdge:
    """서브태스크 간의 의존성 간선"""
    from_id: int
    to_id: int
    dependency_type: str 
    reason: str = ""

@dataclass
class SubtaskDAG:
    """전체 작업(Task) 내 서브태스크들의 전체 DAG 구조"""
    task_name: str
    nodes: List[SubtaskSummary] = field(default_factory=list)
    edges: List[SubtaskEdge] = field(default_factory=list)
    parallel_groups: Dict[int, List[int]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "task_name": self.task_name,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "parallel_groups": self.parallel_groups
        }


class DAGGenerator:
    """LLM 기반 DAG 생성기 (Action DAG + Subtask DAG)"""

    def __init__(self, api_key_file: str = "api_key", gpt_version: str = "gpt-4o"):
        self.gpt_version = gpt_version
        self._setup_api(api_key_file)

    def _setup_api(self, api_key_file: str) -> None:
        """
        api_key_file:
        - "api_key" or "api_key.txt" 같은 이름을 받아도 됨
        실제 위치:
        - AI2Thor/baselines/PDL/api_key.txt (이 DAG_Module.py와 같은 폴더 기준)
        """
        here = Path(__file__).resolve().parent  # .../AI2Thor/baselines/PDL/scripts
        pdl_dir = here.parent                   # .../AI2Thor/baselines/PDL

        candidates = [
            Path(api_key_file).expanduser(),                 # 혹시 절대/상대경로로 넘겼을 때
            Path(api_key_file + ".txt").expanduser(),
            pdl_dir / api_key_file,                          
            pdl_dir / (api_key_file + ".txt"),
        ]

        for c in candidates:
            if c.exists() and c.is_file():
                openai.api_key = c.read_text().strip()
                print(f"[DAG] Loaded API key from: {c}")
                return

        raise FileNotFoundError(f"[DAG] api key file not found. Tried: {[str(c) for c in candidates]}")

    def parse_action(self, action_str: str) -> Tuple[str, str, List[str]]:
        """PDDL 액션 문자열을 파싱 (액션타입, 로봇, 물체들 추출)"""
        action_str = re.sub(r'\s*\(\d+\)\s*$', '', action_str).strip()
        parts = action_str.split()
        if len(parts) < 2:
            return (action_str, '', [])
        action_type = parts[0].lower()
        robot = parts[1]
        objects = parts[2:] if len(parts) > 2 else []
        return (action_type, robot, objects)

    # -------------------------
    # 1) Action-level DAG 프롬프트 및 분석
    # -------------------------
    def _create_action_dag_prompt(self, plan_actions: List[str], problem_content: str, precondition_content: str) -> str:
        actions_numbered = "\n".join([f"{i}: {a}" for i, a in enumerate(plan_actions)])

        # few-shot 예시를 prompt에 포함 (바인딩 잘 뽑게)
        fewshot = (
            "### Example 1\n"
            "Actions:\n"
            "0: gotoobject robot? apple\n"
            "1: pickupobject robot? apple countertop\n"
            "2: gotoobject robot? fridge\n"
            "3: openfridge robot? fridge\n"
            "4: putobjectinfridge robot? apple fridge\n"
            "Correct dependencies:\n"
            "- causal: 0 -> 1\n"
            "- causal: 1 -> 2\n"
            "- causal: 2 -> 3\n"
            "- causal: 3 -> 4\n"
            "- binding: 1 -> 4 (apple is held and must be carried by the same robot)\n\n"
            "### Example 2\n"
            "Actions:\n"
            "0: gotoobject robot? fridge\n"
            "1: openfridge robot? fridge\n"
            "2: closefridge robot? fridge\n"
            "Correct dependencies:\n"
            "- causal: 0 -> 1\n"
            "- causal: 1 -> 2\n"
            "- No binding\n\n"
        )

        prompt = (
            "You are a PDDL plan dependency analyzer.\n\n"
            "IMPORTANT CONTEXT:\n"
            "- This plan is generated BEFORE task-to-robot assignment.\n"
            "- Robot labels (e.g., robot1) are PLACEHOLDERS and must NOT constrain parallelism.\n"
            "- However, some dependencies require SAME ROBOT due to robot-bound state (holding/carrying).\n\n"
            f"{fewshot}"
            f"## PDDL Problem:\n{problem_content}\n\n"
            f"## Precondition Information:\n{precondition_content}\n\n"
            f"## Plan Actions (numbered):\n{actions_numbered}\n\n"
            "## Task:\n"
            "Return ALL necessary dependencies between actions. Each dependency MUST be labeled as one of:\n"
            "1) causal: an action produces a precondition required by another action.\n"
            "2) resource: actions manipulate the same physical object so they cannot be parallel.\n"
            "3) binding: SAME ROBOT REQUIRED because holding/carrying state must persist.\n"
            "   - Typical: pickupobject -> putobject/putobjectinfridge/throwobject/drophandobject for SAME object.\n"
            "   - Even if not adjacent, if object must remain held across intermediate actions, add a binding edge.\n\n"
            "## Output Format (JSON only):\n"
            "{\n"
            '  "dependencies": [\n'
            '    {"from": <int>, "to": <int>, "type": "<causal|resource|binding>", "reason": "<brief>"}\n'
            "  ]\n"
            "}\n\n"
            "IMPORTANT:\n"
            "- Output ONLY valid JSON. No markdown.\n"
        )
        return prompt

    def analyze_action_dependencies(self, plan_actions: List[str], problem_content: str = "", precondition_content: str = "") -> Dict:
        """LLM을 호출하여 액션 간 의존성을 JSON으로 받아옴"""
        prompt = self._create_action_dag_prompt(plan_actions, problem_content, precondition_content)
        try:
            response = openai.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "You are a PDDL expert. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0
            )
            result_text = response.choices[0].message.content.strip()

            if "```json" in result_text:
                result_text = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL).group(1)
            elif "```" in result_text:
                result_text = re.search(r'```\s*(.*?)\s*```', result_text, re.DOTALL).group(1)

            return json.loads(result_text)
        except Exception as e:
            print(f"[DAG] LLM action analysis error: {e}")
            return {"dependencies": []}

    def compute_parallel_groups(self, n_nodes: int, edges: List[DAGEdge]) -> Dict[int, List[int]]:
        """위상 정렬(Topological Sort)을 기반으로 동시에 실행 가능한 그룹(레벨) 계산"""
        preds = {i: set() for i in range(n_nodes)}
        succs = {i: set() for i in range(n_nodes)}

        for e in edges:
            # 모든 의존성 타입(causal, resource, binding)을 순서 제약으로 반영
            u, v = e.from_id, e.to_id
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                preds[v].add(u)
                succs[u].add(v)

        from collections import deque
        indeg = {i: len(preds[i]) for i in range(n_nodes)}
        level = {i: 0 for i in range(n_nodes)}
        q = deque([i for i in range(n_nodes) if indeg[i] == 0])

        while q:
            u = q.popleft()
            for v in succs[u]:
                level[v] = max(level[v], level[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        groups: Dict[int, List[int]] = {}
        for i in range(n_nodes):
            g = level[i]
            groups.setdefault(g, []).append(i)
        return groups

    def build_dag(self, subtask_name: str, plan_actions: List[str],
                  problem_content: str = "", precondition_content: str = "") -> PlanDAG:
        """실제 액션 노드와 간선을 생성하여 PlanDAG 객체 구축"""
        # 1. 노드 생성
        nodes: List[ActionNode] = []
        for i, action in enumerate(plan_actions):
            action_type, robot, objects = self.parse_action(action)
            nodes.append(ActionNode(
                id=i, action=action, action_type=action_type, robot=robot, objects=objects
            ))

        n = len(nodes)

        # 2. LLM 의존성 분석 호출
        analysis = self.analyze_action_dependencies(plan_actions, problem_content, precondition_content)

        # 3. 간선(Edge) 생성
        edges: List[DAGEdge] = []
        seen = set()  # (from,to,type)
        dropped = 0

        for dep in analysis.get("dependencies", []):
            try:
                u = int(dep.get("from"))
                v = int(dep.get("to"))
            except Exception:
                dropped += 1
                continue

            dep_type = (dep.get("type") or "").lower().strip()
            if dep_type not in ("causal", "resource", "binding"):
                dep_type = "causal"

            # 범위 밖이면 버림 (유령 노드 제거)
            if not (0 <= u < n and 0 <= v < n):
                dropped += 1
                continue

            key = (u, v, dep_type)
            if key in seen:
                continue
            seen.add(key)

            edges.append(DAGEdge(from_id=u, to_id=v, dependency_type=dep_type))

        if dropped:
            print(f"[DAG] Dropped {dropped} invalid dependencies (out-of-range or malformed)")

        # 4. 병렬 그룹(단계) 계산
        parallel_groups = self.compute_parallel_groups(len(nodes), edges)
        for g, node_ids in parallel_groups.items():
            for nid in node_ids:
                nodes[nid].parallel_group = g

        return PlanDAG(
            subtask_name=subtask_name,
            nodes=nodes,
            edges=edges,
            parallel_groups=parallel_groups
        )

    def save_dag_json(self, dag: PlanDAG, output_path: str) -> None:
        with open(output_path, 'w') as f:
            json.dump(dag.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[DAG] Saved JSON to {output_path}")

    def visualize_dag(self, dag: PlanDAG, output_path: str) -> None:
        """pydot(Graphviz)을 사용하여 DAG를 PNG/PDF 이미지로 시각화"""
        try:
            import pydot
        except ImportError:
            print("[DAG] pydot not installed. Try: pip install pydot graphviz")
            return

        graph = pydot.Dot(graph_type="digraph", rankdir="TB", splines="spline", bgcolor="white")
        graph.set_node_defaults(shape="box", style="rounded,filled", fillcolor="white",
                                fontname="Helvetica", fontsize="10", margin="0.08,0.05")
        graph.set_edge_defaults(color="#555555", arrowsize="0.7", penwidth="1.2")

        group_nodes: Dict[int, List[str]] = {}
        for node in dag.nodes:
            label = f"{node.id}. {node.action_type}\\n[{', '.join(node.objects)}]"
            graph.add_node(pydot.Node(str(node.id), label=label))
            group_nodes.setdefault(node.parallel_group, []).append(str(node.id))

        for g, ids in sorted(group_nodes.items()):
            sg = pydot.Cluster(
                graph_name=f"cluster_g{g}",
                label=f"병렬실행 Group {g}",
                color="#DDDDDD",
                style="rounded",
                fontname="Helvetica",
                fontsize="11",
                penwidth="1"
            )
            sg.add_node(pydot.Node("rank_dummy_" + str(g), style="invis"))
            for nid in ids:
                sg.add_node(pydot.Node(nid))
            sg.set_rank("same")
            graph.add_subgraph(sg)

        for edge in dag.edges:
            et = (edge.dependency_type or "").lower()
            if et == "resource":
                style, color, penwidth = "dashed", "#757575", "1.2"
            elif et == "binding":
                style, color, penwidth = "bold", "#E53935", "2.5"
            else:
                style, color, penwidth = "solid", "#333333", "1.2"

            graph.add_edge(pydot.Edge(
                str(edge.from_id), str(edge.to_id),
                style=style, color=color, penwidth=penwidth
            ))

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".pdf":
            graph.write_pdf(output_path)
        else:
            graph.write_png(output_path)

        print(f"[DAG] Saved pretty visualization to {output_path}")

    # ==========================================================
    # 2) Subtask-level DAG (서브태스크 간 관계 분석)
    # ==========================================================

    def _create_subtask_summary_prompt(self, subtask_name: str, plan_actions: List[str],
                                       problem_content: str, precondition_content: str) -> str:
        actions_txt = "\n".join([f"- {a}" for a in plan_actions])

        prompt = (
            "You are a robotics task abstraction module.\n"
            "Given one SUBTASK's PDDL info and action plan, extract a compact summary.\n\n"
            f"SUBTASK NAME: {subtask_name}\n\n"
            f"PDDL Problem:\n{problem_content}\n\n"
            f"Precondition Info:\n{precondition_content}\n\n"
            f"Plan Actions:\n{actions_txt}\n\n"
            "Return JSON only with the following schema:\n"
            "{\n"
            '  "objects": ["..."],\n'
            '  "preconds": ["..."],\n'
            '  "effects": ["..."]\n'
            "}\n\n"
            "Rules:\n"
            "- objects: include key objects mentioned (e.g., apple, fridge).\n"
            "- preconds: minimal natural-language conditions required before starting.\n"
            "- effects: minimal natural-language outcomes after completion.\n"
            "- Output ONLY valid JSON.\n"
        )
        return prompt

    def build_subtask_summary(self, subtask_id: int, subtask_name: str,
                             plan_actions: List[str], problem_content: str, precondition_content: str) -> SubtaskSummary:
        prompt = self._create_subtask_summary_prompt(subtask_name, plan_actions, problem_content, precondition_content)
        """LLM을 통해 서브태스크가 '무엇을 위해 하는 일인지' 요약 정보를 추출"""
        try:
            response = openai.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            if "```json" in text:
                text = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL).group(1)
            elif "```" in text:
                text = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL).group(1)

            data = json.loads(text)
            return SubtaskSummary(
                id=subtask_id,
                name=subtask_name,
                objects=data.get("objects", []) or [],
                preconds=data.get("preconds", []) or [],
                effects=data.get("effects", []) or []
            )
        except Exception as e:
            print(f"[Subtask] summary LLM error for {subtask_name}: {e}")
            # fallback: objects는 action에서 대충
            objs = []
            for a in plan_actions:
                _, _, o = self.parse_action(a)
                objs.extend(o)
            objs = sorted(list({x for x in objs if x}))
            return SubtaskSummary(id=subtask_id, name=subtask_name, objects=objs, preconds=[], effects=[])

    def _create_subtask_dag_prompt(self, task_name: str, summaries: List[SubtaskSummary]) -> str:
        # summaries를 텍스트로 풀어서 제공
        lines = []
        for s in summaries:
            lines.append(
                f"{s.id}: name={s.name}\n"
                f"   objects={s.objects}\n"
                f"   preconds={s.preconds}\n"
                f"   effects={s.effects}\n"
            )
        summaries_txt = "\n".join(lines)

        # few-shot: subtask dependency 예시
        fewshot = (
            "### Example A (HAS dependency: clear causal link)\n"
            "0: preconds=['container is closed'], effects=['container is open']\n"
            "1: preconds=['container is open'], effects=['item stored']\n"
            "Output:\n"
            "{\n"
            '  "dependencies": [\n'
            '    {"from": 0, "to": 1, "type": "causal", "reason": "Subtask 1 requires container to be open"}\n'
            "  ],\n"
            '  "parallel_groups": [[0], [1]]\n'
            "}\n\n"

            "### Example B (NO dependency: completely different objects)\n"
            "0: objects=['document'], effects=['document archived']\n"
            "1: objects=['switch'], effects=['switch turned off']\n"
            "Output:\n"
            "{\n"
            '  "dependencies": [],\n'
            '  "parallel_groups": [[0, 1]]\n'
            "}\n\n"

            "### Example C (HAS dependency: SAME OBJECT manipulated sequentially)\n"
            "0: name='Wash the Fork', objects=['fork', 'sink'], effects=['fork is clean', 'fork at sink']\n"
            "1: name='Put Fork in Bowl', objects=['fork', 'bowl'], preconds=['fork is clean'], effects=['fork in bowl']\n"
            "Output:\n"
            "{\n"
            '  "dependencies": [\n'
            '    {"from": 0, "to": 1, "type": "causal", "reason": "Fork must be washed before putting in bowl"},\n'
            '    {"from": 0, "to": 1, "type": "resource", "reason": "Same fork object - cannot be manipulated in parallel"}\n'
            "  ],\n"
            '  "parallel_groups": [[0], [1]]\n'
            "}\n\n"

            "### Example D (HAS dependency: object state change)\n"
            "0: objects=['apple'], effects=['robot holding apple']\n"
            "1: objects=['apple', 'fridge'], preconds=['robot holding apple'], effects=['apple in fridge']\n"
            "Output:\n"
            "{\n"
            '  "dependencies": [\n'
            '    {"from": 0, "to": 1, "type": "causal", "reason": "Must hold apple before putting in fridge"},\n'
            '    {"from": 0, "to": 1, "type": "binding", "reason": "Apple must be carried by same robot"}\n'
            "  ],\n"
            '  "parallel_groups": [[0], [1]]\n'
            "}\n\n"
        )

        prompt = (
            "You are a multi-agent task planner.\n"
            "Given a list of SUBTASK summaries, determine subtask-level dependencies.\n"
            "Robot labels like 'robot1' are placeholders and MUST NOT create dependencies by themselves.\n\n"
            f"TASK NAME: {task_name}\n\n"
            f"{fewshot}"
            "SUBTASK SUMMARIES:\n"
            f"{summaries_txt}\n\n"
            "Return JSON only with schema:\n"
            "{\n"
            '  "dependencies": [\n'
            '    {"from": <int>, "to": <int>, "type": "<causal|resource|binding|ordering>", "reason": "<brief>"},\n'
            "    ...\n"
            "  ],\n"
            '  "parallel_groups": [\n'
            "    [<subtask_ids parallel>],\n"
            "    [<next step>]\n"
            "  ]\n"
            "}\n\n"
            "DEPENDENCY DETECTION RULES (IMPORTANT!):\n"
            "1. SHARED OBJECT = DEPENDENCY: If two subtasks manipulate the SAME physical object (e.g., both use 'fork'),\n"
            "   they CANNOT run in parallel. Add a 'resource' dependency.\n"
            "2. CAUSAL: If subtask B's precondition mentions a state that subtask A's effect produces\n"
            "   (e.g., A: 'robot holding X', B needs 'robot holding X'), add 'causal' dependency.\n"
            "3. BINDING: If an object must be held/carried across subtasks (pickup in A, use in B), add 'binding'.\n"
            "4. SEQUENTIAL OPERATIONS: Actions like 'wash X then put X somewhere' are ALWAYS sequential.\n\n"
            "RULES FOR NO DEPENDENCY:\n"
            "- Only return empty dependencies if subtasks use COMPLETELY DIFFERENT objects.\n"
            "- Generic robot states ('robot not inaction') do NOT create dependencies.\n\n"
            "Output ONLY valid JSON. No markdown.\n"
        )
        return prompt

    def build_subtask_dag(self, task_name: str, summaries: List[SubtaskSummary]) -> SubtaskDAG:
        prompt = self._create_subtask_dag_prompt(task_name, summaries)
        """서브태스크 요약본을 비교하여 전체적인 실행 순서(Subtask DAG) 구축"""
        try:
            response = openai.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            if "```json" in text:
                text = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL).group(1)
            elif "```" in text:
                text = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL).group(1)

            data = json.loads(text)

            edges: List[SubtaskEdge] = []
            for d in data.get("dependencies", []):
                t = (d.get("type") or "").lower().strip()
                if t not in ("causal", "resource", "binding", "ordering"):
                    t = "causal"
                edges.append(SubtaskEdge(
                    from_id=int(d["from"]),
                    to_id=int(d["to"]),
                    dependency_type=t,
                    reason=d.get("reason", "")
                ))

            # parallel_groups가 없으면 causal만으로 레벨 계산해서 만들기
            pg = {}
            raw_pg = data.get("parallel_groups")
            if isinstance(raw_pg, list) and raw_pg:
                for gi, g in enumerate(raw_pg):
                    pg[gi] = [int(x) for x in g]
            else:
                # causal + ordering을 스케줄 제약으로 사용 (실제 node ID 사용)
                node_ids = [s.id for s in summaries]
                pg = self._compute_subtask_parallel_groups(node_ids, edges)

            return SubtaskDAG(
                task_name=task_name,
                nodes=summaries,
                edges=edges,
                parallel_groups=pg
            )
        except Exception as e:
            print(f"[Subtask] DAG LLM error: {e}")
            # fallback: 빈 DAG
            return SubtaskDAG(task_name=task_name, nodes=summaries, edges=[], parallel_groups={0: [s.id for s in summaries]})

    def _compute_subtask_parallel_groups(self, node_ids: List[int], edges: List[SubtaskEdge]) -> Dict[int, List[int]]:
        # causal/order만 순서 제약으로 사용
        # node_ids는 실제 subtask ID 리스트 (1-based일 수 있음)
        id_set = set(node_ids)
        preds = {i: set() for i in node_ids}
        succs = {i: set() for i in node_ids}

        for e in edges:
            # 모든 의존성 타입(causal, resource, binding, ordering)을 순서 제약으로 반영
            u, v = e.from_id, e.to_id
            if u in id_set and v in id_set:
                preds[v].add(u)
                succs[u].add(v)

        from collections import deque
        indeg = {i: len(preds[i]) for i in node_ids}
        level = {i: 0 for i in node_ids}
        q = deque([i for i in node_ids if indeg[i] == 0])

        while q:
            u = q.popleft()
            for v in succs[u]:
                level[v] = max(level[v], level[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        groups: Dict[int, List[int]] = {}
        for i in node_ids:
            groups.setdefault(level[i], []).append(i)
        return groups

    def save_subtask_dag_json(self, dag: SubtaskDAG, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(dag.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Subtask] Saved Subtask DAG JSON to {output_path}")

    def visualize_subtask_dag(self, dag: SubtaskDAG, output_path: str) -> None:
        try:
            import pydot
        except ImportError:
            print("[Subtask] pydot not installed. Try: pip install pydot graphviz")
            return

        graph = pydot.Dot(graph_type="digraph", rankdir="TB", splines="spline", bgcolor="white")
        graph.set_node_defaults(shape="box", style="rounded,filled", fillcolor="white",
                                fontname="Helvetica", fontsize="10", margin="0.08,0.05")
        graph.set_edge_defaults(color="#555555", arrowsize="0.7", penwidth="1.2")

        group_nodes: Dict[int, List[str]] = {}
        node_to_group = {}
        for g, ids in dag.parallel_groups.items():
            for sid in ids:
                node_to_group[sid] = g

        for s in dag.nodes:
            gid = node_to_group.get(s.id, 0)
            label = f"{s.id}. {s.name}\\nobjs={len(s.objects)}"
            graph.add_node(pydot.Node(str(s.id), label=label))
            group_nodes.setdefault(gid, []).append(str(s.id))

        for g, ids in sorted(group_nodes.items()):
            sg = pydot.Cluster(
                graph_name=f"cluster_sub_g{g}",
                label=f"Subtask Group {g}",
                color="#DDDDDD",
                style="rounded",
                fontname="Helvetica",
                fontsize="11",
                penwidth="1"
            )
            sg.add_node(pydot.Node("rank_dummy_sub_" + str(g), style="invis"))
            for nid in ids:
                sg.add_node(pydot.Node(nid))
            sg.set_rank("same")
            graph.add_subgraph(sg)

        for e in dag.edges:
            t = (e.dependency_type or "").lower()
            if t == "resource":
                style, color, penwidth = "dashed", "#757575", "1.2"
            elif t == "binding":
                style, color, penwidth = "bold", "#E53935", "2.5"
            elif t == "ordering":
                style, color, penwidth = "dotted", "#4E6CEF", "1.4"
            else:  # causal
                style, color, penwidth = "solid", "#333333", "1.2"

            graph.add_edge(pydot.Edge(
                str(e.from_id), str(e.to_id),
                style=style, color=color, penwidth=penwidth
            ))

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".pdf":
            graph.write_pdf(output_path)
        else:
            graph.write_png(output_path)

        print(f"[Subtask] Saved Subtask DAG visualization to {output_path}")


class DAGProcessor:
    """프로젝트 폴더 내의 모든 PDDL 계획 파일을 읽어 DAG를 생성하는 메인 프로세서"""

    def __init__(self, base_path: str, api_key_file: str = "api_key", gpt_version: str = "gpt-4o"):
        self.base_path = base_path
        self.generator = DAGGenerator(api_key_file, gpt_version)

        # 모든 생성 파일 주소 연결
        self.resources_path = os.path.join(base_path, "resources")
        self.plans_path = os.path.join(self.resources_path, "subtask_pddl_plans")
        self.problems_path = os.path.join(self.resources_path, "subtask_pddl_problems")
        self.preconditions_path = os.path.join(self.resources_path, "precondition_subtasks")
        self.output_path = os.path.join(self.resources_path, "dag_outputs")

        os.makedirs(self.output_path, exist_ok=True)

    def find_matching_files(self, plan_file: str) -> Tuple[Optional[str], Optional[str]]:
        base_name = plan_file.replace("_actions.txt", "").replace("_allocated.txt", "")
        problem_file = os.path.join(self.problems_path, f"{base_name}.pddl")
        precond_file = os.path.join(self.preconditions_path, base_name.replace("subtask_", "pre_") + ".txt")
        problem_path = problem_file if os.path.exists(problem_file) else None
        precond_path = precond_file if os.path.exists(precond_file) else None
        return problem_path, precond_path

    def process_all_plans(self, task_name: str = "task") -> Tuple[List[PlanDAG], SubtaskDAG]:
        """모든 서브태스크 계획에 대해 Action DAG와 Subtask DAG를 일괄 생성"""

        action_dags: List[PlanDAG] = []
        summaries: List[SubtaskSummary] = []

        #모든 PDDL plan 불러오기
        plan_files = [f for f in os.listdir(self.plans_path) if f.endswith("_actions.txt")]

        for idx, plan_file in enumerate(sorted(plan_files)):
            print(f"\n[DAG] Processing: {plan_file}")

            plan_path = os.path.join(self.plans_path, plan_file)
            with open(plan_path, 'r') as f:
                plan_actions = [line.strip() for line in f.readlines() if line.strip()]

            if not plan_actions:
                print(f"[DAG] Empty plan file: {plan_file}")
                continue

            problem_path, precond_path = self.find_matching_files(plan_file)
            problem_content = ""
            precond_content = ""

            if problem_path:
                with open(problem_path, 'r') as f:
                    problem_content = f.read()

            if precond_path:
                with open(precond_path, 'r') as f:
                    precond_content = f.read()

            subtask_name = plan_file.replace("_actions.txt", "")

            # 1) Action DAG 만들기
            dag = self.generator.build_dag(subtask_name, plan_actions, problem_content, precond_content)
            action_dags.append(dag)

            json_output = os.path.join(self.output_path, f"{subtask_name}_dag.json")
            self.generator.save_dag_json(dag, json_output)

            img_output = os.path.join(self.output_path, f"{subtask_name}_dag.png")
            self.generator.visualize_dag(dag, img_output)

            # 2) Subtask Summary (LLM) 생성
            s = self.generator.build_subtask_summary(
                subtask_id=len(summaries),
                subtask_name=subtask_name,
                plan_actions=plan_actions,
                problem_content=problem_content,
                precondition_content=precond_content
            )
            summaries.append(s)

        # 3) Subtask DAG (LLM) 만들기
        subtask_dag = self.generator.build_subtask_dag(task_name=task_name, summaries=summaries)

        subtask_json = os.path.join(self.output_path, f"{task_name}_SUBTASK_DAG.json")
        self.generator.save_subtask_dag_json(subtask_dag, subtask_json)

        subtask_img = os.path.join(self.output_path, f"{task_name}_SUBTASK_DAG.png")
        self.generator.visualize_subtask_dag(subtask_dag, subtask_img)

        return action_dags, subtask_dag

    def get_execution_schedule(self, dag: PlanDAG) -> List[List[str]]:
        schedule = []
        for group_idx in sorted(dag.parallel_groups.keys()):
            node_ids = dag.parallel_groups[group_idx]
            actions = [dag.nodes[nid].action for nid in node_ids if nid < len(dag.nodes)]
            schedule.append(actions)
        return schedule


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate DAG from PDDL plans")
    parser.add_argument("--base-path", type=str, default=os.getcwd(), help="Base path of the project")
    parser.add_argument("--api-key-file", type=str, default="api_key", help="OpenAI API key file")
    parser.add_argument("--gpt-version", type=str, default="gpt-4o", help="GPT model version")
    parser.add_argument("--task-name", type=str, default="task", help="Name for subtask-level DAG output files")
    args = parser.parse_args()

    processor = DAGProcessor(base_path=args.base_path, api_key_file=args.api_key_file, gpt_version=args.gpt_version)

    action_dags, subtask_dag = processor.process_all_plans(task_name=args.task_name)

    print(f"\n[DAG] Processed {len(action_dags)} subtasks")
    print(f"[Subtask] Built subtask DAG with {len(subtask_dag.nodes)} nodes and {len(subtask_dag.edges)} edges")

    for dag in action_dags:
        print(f"\n=== {dag.subtask_name} ===")
        schedule = processor.get_execution_schedule(dag)
        for i, group in enumerate(schedule):
            print(f"  Step {i} (parallel): {group}")


if __name__ == "__main__":
    main()