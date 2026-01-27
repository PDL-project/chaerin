import copy
import glob
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import time
import re
import shutil
import sys
from typing import List, Dict, Tuple, Optional, Union, Any

import openai
import ai2thor.controller

import difflib
import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots

DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0
DEFAULT_RETRY_DELAY = 20
MAX_RETRIES = 3

class PDDLError(Exception):
    """Base exception class for PDDL-related errors."""
    pass

class PDDLUtils:
    """
    AI2-THOR 환경에서 객체 불러와서, llm이나 pddl이 활용하기 좋은 형태로 변환해주는 namespace
    """
    
    @staticmethod #인스턴트 생성x, 그냥 사용하면 됨, 클래스 내부 상태와 상관없이 실행되는 함수라는 의미
    def convert_to_dict_objprop(objs: List[str], obj_mass: List[float]) -> List[Dict[str, Union[str, float]]]:
        """
        객체 리스트(사과, 칼), 각 객체의 질량(0.2, 0.5)을 zip으로 이름과 질량을 묶어서 딕셔너리 리스트로 변환해주는 함수

        llm이나, pddl에 넣기 좋은 형태가 딕셔너리 리스트라 그걸로 바꿔주기 위한 함수다!
        """
        return [{'name': obj, 'mass': mass} for obj, mass in zip(objs, obj_mass)]
    
    @staticmethod
    def get_ai2_thor_objects(floor_plan: int) -> List[Dict[str, Any]]:
        """
        입력 : floor_plan -> 15, 201 같은 데이터셋 숫자

        출력 : objects_ai가 될 형식임
        [
            {'name': 'Apple', 'mass': 0.2},
            {'name': 'Knife', 'mass': 0.5},
            {'name': 'Fridge', 'mass': 50.0},
            ...
        ] 
        """
        controller = None
        try:
            controller = ai2thor.controller.Controller(scene=f"FloorPlan{floor_plan}")
            objects_ai = []

            for obj in controller.last_event.metadata["objects"]:
                name = obj["objectType"]
                mass = obj.get("mass", 0.0)
                parents = obj.get("parentReceptacles")
                if parents:
                    locations = [p.split("|")[0] for p in parents]
                else:
                    locations = ["Floor"]

                objects_ai.append({
                    "name": name,
                    "mass": mass,
                    "locations": locations
                })

            return objects_ai

        finally:
            if controller:
                controller.stop()

class FileProcessor:
    """
    파일 입출력
    PDDL 텍스트 정리 매니저

    LLM이 만든 큰 PDDL 텍스트를 서브태스크별 .pddl 파일로 쪼개서 저장
    도메인 이름 추출, planner 출력에서 plan만 뽑기, 폴더 청소, BDDL 파싱
    """
    
    def __init__(self, base_path: str):
        """file processor 생성자
        
        Args:
            base_path (str): 기본 루트
        """
        self.base_path = base_path
        self.subtask_path = os.path.join(base_path, "resources", "generated_subtask")#LLM이 만든 pddl problem을 쪼개서 저장하는 곳(검증전)
        self.validated_subtask_path = os.path.join(base_path, "resources", "validated_subtask")  #pddl problem을 검증한걸 저장하는 곳(검증후)
        self.each_run_path = os.path.join(base_path, "resources", "each_run") # 매 실행할때마다 실행별 기록을 저장하는 곳

        self.subtask_pddl_problems_path = os.path.join(base_path, "resources", "subtask_pddl_problems") #각 서브테스크에 대한 pddl problems 저장해놓는 곳
        os.makedirs(self.subtask_pddl_problems_path, exist_ok=True)

        self.subtask_pddl_plans_path = os.path.join(base_path, "resources", "subtask_pddl_plans") #각 서브테스크에 대한 pddl plans 저장해놓는 곳
        os.makedirs(self.subtask_pddl_plans_path, exist_ok=True)

        self.precondition_subtasks_path = os.path.join(base_path, "resources", "precondition_subtasks") #각 서브테스크에 대한 pddl만들기 위한 사전컨디션 규칙 정해놓은 파일 저장해놓는 곳
        os.makedirs(self.precondition_subtasks_path, exist_ok=True)

        os.makedirs(self.subtask_path, exist_ok=True)
        os.makedirs(self.validated_subtask_path, exist_ok=True) 
        os.makedirs(self.each_run_path, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """
        텍스트 파일을 읽어서 문자열로 반환 함수
        """
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            raise PDDLError(f"파일을 찾을수없음: {file_path}")
        except Exception as e:
            raise PDDLError(f"파일을 읽을수없음 {file_path}: {str(e)}")
    
    def write_file(self, file_path: str, content: str) -> None:
        """
        문자열을 파일로 저장
        """
        try:
            with open(file_path, 'w') as file:
                file.write(content)
        except Exception as e:
            raise PDDLError(f"파일 생성 실패 {file_path}: {str(e)}")
    
    def split_pddl_tasks(self, code_plan: Union[str, List[str]], isValidated: bool) -> None:
        """llm이 한번에 생성한 여러개의 pddl을 분리해서 저장하는 함수
        
        Args:
            code_plan (List[str]): List of PDDL plans to split
        """
        try:
            # code_plan이 문자열이면 리스트로 바꾸기
            if isinstance(code_plan, str):
                code_plan = [code_plan]
            
            # 실행시각 기준으로, resources/each_run/<timestamp>/ 폴더 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_directory = os.path.join(self.each_run_path, timestamp)
            os.makedirs(timestamp_directory, exist_ok=True)
            
            for i, plan in enumerate(code_plan): #plan 텍스트를 (define (problem 기준으로 split
                tasks = re.split(r"\s*\(define\s*\(problem", plan)
                
                for j, task in enumerate(tasks[1:]):
                    task = "(define (problem" + task
                    task = self.balance_parentheses(task)
                    
                    match = re.search(r'\(problem\s+(\w+)\)', task, re.IGNORECASE)
                    if match:
                        task_name = match.group(1)
                        filename = f"{i+1}_{j+1}_{task_name}.pddl"
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{i+1}_{j+1}_{timestamp}.pddl"
                    
                    # 생성된거 파일에 저장
                    filepath = os.path.join(timestamp_directory, filename)
                    self.write_file(filepath, task)
                    
                    # 저장 위치 (isValidated=False → generated_subtask/, isValidated=True → validated_subtask/)
                    if isValidated:
                        subtask_filepath = os.path.join(self.validated_subtask_path, filename) 
                    else:
                        subtask_filepath = os.path.join(self.subtask_path, filename)
                    print("pddl problem 파일 생성됨:", subtask_filepath) 
                    self.write_file(subtask_filepath, task)
            
        except Exception as e:
            raise PDDLError(f"PDDL tasks 분리 과정에서 에러남: {str(e)}")
    
    def normalize_pddl(self, text: str) -> str: #이거 함수 지우면 안댐!@@@
        if not text:
            return ""
        text = text.replace("```pddl", "").replace("```", "").replace("`", "")
        # (define부터 시작하도록 자르기
        i = text.find("(define")
        if i != -1:
            text = text[i:]
        # '#'로 시작하는 라인 제거
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
        return "\n".join(lines).strip()

    def balance_parentheses(self, content: str) -> str:
        """
        LLM이 만든 pddl이 종종 괄호가 깨져서, 첫번째 완전한 괄호 덩어리만 잘라서 반환해주는 함수
        """
        open_count = 0
        start_index = -1
        end_index = -1
        
        for i, char in enumerate(content):
            if char == '(':
                if open_count == 0:
                    start_index = i
                open_count += 1
            elif char == ')':
                open_count -= 1
                if open_count == 0:
                    end_index = i
                    break
        
        if start_index != -1 and end_index != -1:
            return content[start_index:end_index+1]
        return ""
    
    def split_and_store_tasks(
        self,
        content: str,
        llm: Optional['LLMHandler'] = None,
        gpt_version: Optional[str] = None
    ) -> Tuple[List[str], str]:
        """
        “Problem content summary” 텍스트를 SubTask 단위로 쪼개는 함수

        입력 : LLM이 만든 “problem summary + sequence operations” 텍스트
        출력 : subtasks: List[str] : #SubTask 1: ... 블록 리스트
              sequence_operations: str : “Sequence of Operations:” 뒤 내용
        """
        import re, difflib  

        sequence_operations = ""

        # 1) problem_summary,sequence_operations 추출
        summary_match = re.search(
            r'(?:#?\s*)?Problem\s*content\s*summary\s*:?(.*?)(?=(?:#?\s*)?Sequence\s*of\s*Operations?\s*:?)',
            content, re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            problem_summary = summary_match.group(1).strip()
        else:
            split_marker = re.search(r'(?:#?\s*)?Sequence\s*of\s*Operations?\s*:', content, re.IGNORECASE)
            if split_marker:
                problem_summary = content[:split_marker.start()].strip()
                sequence_operations = content[split_marker.end():].strip()
            else:
                problem_summary = content.strip()
                sequence_operations = "failed to extract2"

        # 2) "#SubTask 1:" / "# SubTask 2:" / "SubTask 3:"
        subtask_block_re = re.compile(
            r'(?im)^\s*#?\s*Sub\s*Task\s*\d+\s*:\s*.*?(?=^\s*#?\s*Sub\s*Task\s*\d+\s*:\s*|\Z)',
            re.DOTALL
        )
        subtasks = [m.group(0).strip() for m in subtask_block_re.finditer(problem_summary)]

        # subtask 가 없으면 그냥 하나를 전체의 subtask로 봄
        if not subtasks:
            subtasks = [problem_summary.strip()] if problem_summary.strip() else []

        print("Subtasks", subtasks) #서브테스크 디버깅
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        # 3) 형식 검증
        # **Assigned Robot**: robot1 , **Objects Involved**: apple, knife 이런거 찾는부분
        fixed_subtasks = []
        structure_ok = re.compile(
            r'\*\*Assigned\s*Robots?\*\*\s*:\s*.*?\n\*\*Objects\s*Involved\*\*\s*:\s*.*',
            re.DOTALL | re.IGNORECASE
        )
        fallback_plain = re.compile(
            r'\bAssigned\s*Robots?\b\s*:\s*.*?\n\bObjects\s*Involved\b\s*:\s*.*',
            re.DOTALL | re.IGNORECASE
        )

        for subtask in subtasks:
            print("Subtask:", subtask) #서브 테스크 디버깅
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            if structure_ok.search(subtask) or fallback_plain.search(subtask):
                fixed_subtasks.append(subtask)
                continue

            #형식 깨졌으면 다시 돌리기 (서브 테스크 생성 형식 깨졌을때 다시 돌리는 부분)
            if llm and gpt_version:
                fix_prompt = (
                    "The following subtask description needs to be reformatted. Please reformat it to strictly follow this structure:\n\n"
                    "#SubTask [number]: [Task Name]\n\n"
                    "# Initial Precondition analyze due to previous subtask:\n"
                    "#1. [precondition description]\n\n"
                    "[Action descriptions with Parameters, Preconditions, and Effects]\n\n"
                    "**Assigned Robot**: [robot number or 'team']\n"
                    "**Objects Involved**: [list of objects]\n\n"
                    "Important formatting rules:\n"
                    "1. Each section must start with the exact headers shown above\n"
                    "2. The order must be: SubTask header, Preconditions, Action descriptions, Assigned Robot, Objects Involved\n"
                    "3. Use '**Assigned Robot**:' and '**Objects Involved**:' exactly as shown with double asterisks\n"
                    "4. Include all action descriptions with their Parameters, Preconditions, and Effects\n"
                    "5. Keep the original action descriptions if they exist\n\n"
                    "Original subtask:\n" + subtask + "\n\n"
                    "Please provide ONLY the reformatted version following the structure above. Do not add any explanations or additional text."
                )

                if "gpt" not in gpt_version:
                    _, fixed_subtask = llm.query_model(
                        fix_prompt, gpt_version, max_tokens=3000, stop=["def"], frequency_penalty=0.30
                    )
                else:
                    messages = [
                        {"role": "system", "content": "You are a Robot PDDL problem Expert. Your task is to reformat subtask descriptions to match a specific structure. Do not add any explanations or additional text."},
                        {"role": "user", "content": fix_prompt}
                    ]
                    _, fixed_subtask = llm.query_model(messages, gpt_version, max_tokens=3000, frequency_penalty=0.4)

                print("=== Testing match on fixed subtask ===") #다시 생성했을때의 서브 테스크 디버깅
                print("Fixed subtask:", repr(fixed_subtask))
                print("Match result:", structure_ok.search(fixed_subtask) or fallback_plain.search(fixed_subtask))

                if structure_ok.search(fixed_subtask) or fallback_plain.search(fixed_subtask):
                    fixed_subtasks.append(fixed_subtask)
                else:
                    print("LLM structure fix failed, using original subtask")
                    #print("\n".join(difflib.ndiff(fixed_subtask.splitlines(), structure_ok.pattern.splitlines())))
                    fixed_subtasks.append(subtask)
            else:
                print("LLM handler not provided, using original subtask")
                fixed_subtasks.append(subtask)

        return fixed_subtasks, sequence_operations
        
    def extract_domain_name(self, problem_file_path: str) -> Optional[str]:
        """
        PDDL problem 파일 안에서 (:domain xxx)를 찾아 xxx를 반환해주는 함수
        """
        try:
            domain_pattern = re.compile(r'\(\s*:domain\s+(\S+)\s*\)')
            content = self.read_file(problem_file_path)
            match = domain_pattern.search(content)
            return match.group(1) if match else None
        except Exception as e:
            print(f"도메인 이름 추출 못함 {problem_file_path}: {str(e)}")
            return None

    def find_domain_file(self, domain_name: str) -> Optional[str]:
        """
        resources/<domain_name>.pddl 파일이 있나 확인해서 path 반환해줌
        """
        try:
            domain_path = os.path.join(self.base_path, "resources", f"{domain_name}.pddl")
            return domain_path if os.path.isfile(domain_path) else None
        except Exception as e:
            print(f"도메인 파일 찾기 실패함 {domain_name}: {str(e)}")
            return None

    def clean_directory(self, directory_path: str) -> None:
        """
        특정 파일 내용 싹 비우는 함수(파일 삭제, 폴더 삭제)
        """
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    def extract_plan_from_output(self, content: str) -> str:
        """
        fast-downward 출력이 긴데, 그 중에서 실제로 plan처럼 생긴 줄만 정규식으로 골라서 합쳐주는 함수
        """
        if not content or not isinstance(content, str):
            raise ValueError("Invalid content provided to extract_plan_from_output")
            
        try:
            plan_pattern = re.compile(r"^\s*\w+\s+\w+\s+\w+\s+\(\d+\)\s*$", re.MULTILINE)
            plan = plan_pattern.findall(content)
            return "\n".join(plan) if plan else ""
        except Exception as e:
            print(f"결과에서 추출할때 에러남: {str(e)}")
            return ""

class LLMHandler:
    """
    LLM한테 입력값 보내고, 답변 받아오는 LLM 담당자 객체
    """
    
    def __init__(self, api_key_file: str):
        """LLMHandle 초기설정
        
        Args:
            api_key_file (str): api_key 있는 파일
        """
        self.setup_api(api_key_file)
    
    def setup_api(self, api_key_file: str) -> None:
        """OpenAI API key 파일 읽기"""
        try:
            
            try:
                api_key = Path(api_key_file + '.txt').read_text().strip()
                if not api_key:
                    raise ValueError("API key file is empty")
                openai.api_key = api_key
                print("Successfully loaded API key from", api_key_file + '.txt')
            except FileNotFoundError:
                # .txt 아닌 파일에 api_key가 있는지 찾는 부분
                try:
                    api_key = Path(api_key_file).read_text().strip()
                    if not api_key:
                        raise ValueError("API key file is empty")
                    openai.api_key = api_key
                    print("Successfully loaded API key from", api_key_file)
                except FileNotFoundError:
                    raise LLMError(f"API key file not found: {api_key_file} or {api_key_file}.txt")
        except Exception as e:
            raise LLMError(f"Error reading API key file: {str(e)}")
    
    #프롬프트 보내고 텍스트 받기 함수
    def query_model(
        self, 
        prompt: Union[str, List[Dict]], 
        gpt_version: str, 
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stop: Optional[List[str]] = None,
        logprobs: Optional[int] = 1,
        frequency_penalty: float = 0
    ) -> Tuple[dict, str]:
        """
        
        Args:
            prompt: 문자열 혹은 dic
            gpt_version: 사용할 모델명
            max_tokens: gpt 답변 최대 토큰 수
            temperature: 랜덤성/창의성(0은 거의 항상 같은 값, 1은 항상 다른값, 실험을 위해 항상 0으로 설정해야함)
            stop: 특정 문자열이 나오면 항상 멈춤
            logprobs: 각 토큰이 얼마나 확률 높게 생성됐는지에 대한 정보
            frequency_penalty: 같은 단어나 구문 반복하지 않도록하는 패널티(0은 반복허용, 커질수록 반복 줄어듬 근데 너무 크면 문장이 부자연스러워짐)
         
        Returns:
            Tuple of (full response object, generated text)
            response, text = query_model(...) / response= ai의 원본 응답 번체, text는 우리가 실제로 쓰는 답변 문자열
            
            text -> pddl생성, 코드생성
            response -> 토큰 사용량 체크 등
        """
        retry_delay = DEFAULT_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES): #api 연결시도(현재 3번으로 설정됨)
            try:
                if "gpt" not in gpt_version: #모델명에 gpt가 없을경우 -> completion 스타일로 생성
                    response = openai.completions.create(
                        model=gpt_version, 
                        prompt=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        stop=stop, 
                        logprobs=logprobs, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response.choices[0].text.strip()
                else: #모델명에 gpt가 없을경우 -> chat 스타일로 생성
                    response = openai.chat.completions.create(
                        model=gpt_version, 
                        messages=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response.choices[0].message.content.strip()
                    
            except openai.RateLimitError: #ai한테 너무 빨리, 많이 요청해서 제한 걸렸을때 예외처리
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise LLMError("Rate limit exceeded")
                
            except (openai.APIError, openai.APITimeoutError) as e: #서버에러 또는 타임아웃 예외처리
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    continue
                raise LLMError(f"API Error after all retries: {str(e)}")
                
            except Exception as e:
                raise LLMError(f"Unexpected error in LLM query: {str(e)}")

class TaskManager:
    """ 
    파이프라인 총괄 매니저, 여러 모듈들 관리
    """
    
    def __init__(self, base_path: str, gpt_version: str, api_key_file: str, prompt_decompse_set: str = "pddl_train_task_decomposesep", prompt_allocation_set: str = "pddl_train_task_allocationsep"):
        """task manager 생성자, 초기 설정
        """
        self.base_path = base_path
        self.gpt_version = gpt_version
        self.prompt_decompse_set = prompt_decompse_set
        self.prompt_allocation_set = prompt_allocation_set
        
        # 핵심 모듈 생성
        self.llm = LLMHandler(api_key_file) #모델 호출 모듈
        self.file_processor = FileProcessor(base_path) # 파일 읽기, 쓰기, pddl 테스크 분리, 정규식 파싱 모듈
        #self.validator = PDDLValidator(self.llm, self.file_processor) # pddl problem 구조 전제조건 검증 모듈
        #self.planner = PDDLPlanner(base_path, self.file_processor) # fast-downward로 plan 생성, 성공률 계산 모듈
        
        # 파일경로 설정, log저장용 폴더 만들기 코드
        self.resources_path = os.path.join(base_path, "resources") # 가져올 리소스 폴더 위치 설정
        self.logs_path = os.path.join(".", "logs")  # 로그 폴더 생성
        os.makedirs(self.logs_path, exist_ok=True)
        
        # subtask 폴더 비우기
        self.clean_all_resources_directories()
        
        # 결과 저장소 초기화
        self.decomposed_plan: List[str] = []
        self.parsed_subtasks: List[List[dict]] = []
        self.precondition_subtasks: List[List[dict]] = []

        self.subtask_pddl_problems: List[List[dict]] = []
        self.subtask_pddl_plans: List[str] = []
        
        self.objects_ai = None

    def clean_all_resources_directories(self) -> None:
        """
        resources/. 아래 생성된 폴더들의 파일들 전부 지우는 함수
        """
        target_dirs = [
            "generated_subtask",
            "precondition_subtasks",
            "subtask_pddl_problems",
            "validated_subtask",
            "subtask_pddl_plans",
        ]

        for dir_name in target_dirs:
            directory = os.path.join(self.resources_path, dir_name)

            try:
                if not os.path.exists(directory):
                    continue

                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"[CLEAN ERROR] Failed to remove {file_path}: {e}")

            except Exception as e:
                print(f"[CLEAN ERROR] Failed to access directory {directory}: {e}")


    def load_dataset(self, test_file: str) -> Tuple[List[str], List[List[dict]], List[str], List[int], List[int]]:
        """
        json파일 읽고, 관련 정보 로드해오기    
        """
        test_tasks = []
        robots_test_tasks = []
        gt_test_tasks = []
        trans_cnt_tasks = []
        min_trans_cnt_tasks = []
        
        try:
            with open(test_file, "r") as f:  #json 파일 한줄씩 읽어오기
                for line in f.readlines():
                    values = list(json.loads(line).values())
                    test_tasks.append(values[0])
                    robots_test_tasks.append(values[1])
                    gt_test_tasks.append(values[2])
                    trans_cnt_tasks.append(values[3])
                    min_trans_cnt_tasks.append(values[4])
            
            # 로봇 ID값을 실제 로봇 정보들로 변환, !!현재 활용가능한 로봇들의 스킬들만 뽑고있음, 로봇들의 개개인 정보는 로드 x
            available_robot_skills = []
            for robots_list in robots_test_tasks:
                skill_set = set()

                for i, r_id in enumerate(robots_list):
                    rob = robots.robots[r_id-1]  # robot26 = {'name': 'robot26',  'skills': ['GoToObject','SliceObject', 'PickupObject'], 'mass' : 100}
                    skill_set.update(rob['skills']) # 'GoToObject' 같은 스킬만 누적
                available_robot_skills.append(sorted(skill_set))
            print(available_robot_skills)
            return test_tasks, available_robot_skills, gt_test_tasks, trans_cnt_tasks, min_trans_cnt_tasks
            
        except FileNotFoundError:
            raise PDDLError(f"Test file not found: {test_file}")
        except json.JSONDecodeError as e:
            raise PDDLError(f"Error parsing JSON in test file: {str(e)}")
        except Exception as e:
            raise PDDLError(f"Error loading dataset: {str(e)}")
    
    def log_results(self, task: str, idx: int, available_robots: List[dict], 
                   gt_test_tasks: List[str], trans_cnt_tasks: List[int], 
                   min_trans_cnt_tasks: List[int], objects_ai: str,
                   bddl_file_path: Optional[str] = None):

        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        task_name = "_".join(task.split()).replace('\n', '')
        folder_name = f"{task_name}_plans_{date_time}"
        log_folder = os.path.join(self.logs_path, folder_name)
        
        #print(f"Creating log folder: {log_folder}")
        os.makedirs(log_folder)
        # === Save subtask PDDL problems ===
        subtask_pddl_dir = os.path.join(log_folder, "subtask_pddl_problems")
        os.makedirs(subtask_pddl_dir, exist_ok=True)

        for item in self.subtask_pddl_problems[idx]:
            sid = item.get("subtask_id", "unknown")
            title = item.get("subtask_title", "untitled")
            pddl_text = item.get("problem_text", "")

            safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
            filename = f"subtask_{sid:02d}_{safe_title}.pddl"

            path = os.path.join(subtask_pddl_dir, filename)
            self.file_processor.write_file(path, pddl_text)
        
        # === Save precondition subtasks (LLM-generated preconditions/goals) ===
        precondition_dir = os.path.join(log_folder, "precondition_subtasks")
        os.makedirs(precondition_dir, exist_ok=True)

        for item in self.precondition_subtasks[idx]:
            sid = item.get("subtask_id", "unknown")
            title = item.get("subtask_title", "untitled")
            pre_text = item.get("pre_goal_text", "")

            safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
            filename = f"pre_{sid:02d}_{safe_title}.txt"

            path = os.path.join(precondition_dir, filename)
            self.file_processor.write_file(path, pre_text)

        # === Save validated subtask PDDL problems (LLM validator outputs) ===
        val_subtask_pddl_dir = os.path.join(log_folder, "val_subtask_pddl_problems")
        os.makedirs(val_subtask_pddl_dir, exist_ok=True)

        src_val_dir = self.file_processor.validated_subtask_path  # resources/validated_subtask
        for file_name in os.listdir(src_val_dir):
            if file_name.endswith(".pddl"):
                src_path = os.path.join(src_val_dir, file_name)
                dst_path = os.path.join(val_subtask_pddl_dir, file_name)
                shutil.copy(src_path, dst_path)

        # === Save subtask PDDL plans (fastdownward outputs) ===
        plan_action_pddl_dir = os.path.join(log_folder, "subtask_pddl_plans")
        os.makedirs(plan_action_pddl_dir, exist_ok=True)

        plans_dir = self.file_processor.subtask_pddl_plans_path  # resources/validated_subtask
        for file_name in os.listdir(plans_dir):
            if file_name.endswith("actions.txt"):
                main_path = os.path.join(plans_dir, file_name)
                sub_path = os.path.join(plan_action_pddl_dir, file_name)
                shutil.copy(main_path, sub_path)

        try:
            
            self._write_plan(log_folder, "decomposed_plan.py", self.decomposed_plan[idx])
            self._write_plan(log_folder, "validated_plan.py", self.validated_plan[idx])
            
            TC, total_subtasks = self.tc[idx], self.total_subtasks[idx]
            print(f"Task {idx + 1} - TC: {TC}, Total Subtasks: {total_subtasks}")


            with open(os.path.join(log_folder, "log.txt"), 'w') as f:
                f.write(task)
                f.write(f"\n\nGPT Version: {self.gpt_version}")
                f.write(f"\n{objects_ai}")
                f.write(f"\nrobots = {available_robots[idx]}")
                f.write(f"\nground_truth = {gt_test_tasks[idx]}")
                f.write(f"\ntrans = {trans_cnt_tasks[idx]}")
                f.write(f"\nmin_trans = {min_trans_cnt_tasks[idx]}")
                f.write(f"\nTotalsuccesssubtask = {TC}")
                f.write(f"\nTotalsubtask = {total_subtasks}")
            
            subtask_folder = os.path.join(log_folder, "generated_subtask")
            os.makedirs(subtask_folder)
            source_folder = os.path.join(self.resources_path, "generated_subtask")
            for file_name in os.listdir(source_folder):
                full_file_name = os.path.join(source_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, subtask_folder)
            
            validated_subtask_folder = os.path.join(log_folder, "validated_subtask")
            os.makedirs(validated_subtask_folder)
            source_validated_folder = os.path.join(self.resources_path, "validated_subtask")
            for file_name in os.listdir(source_validated_folder):
                full_file_name = os.path.join(source_validated_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, validated_subtask_folder)

        except Exception as e:
            print(f"Error writing plans for task {idx + 1}: {str(e)}")

    def _write_plan(self, folder: str, filename: str, content: Union[str, List]):
        """Write a plan to a file."""
        if isinstance(content, list):
            for i, item in enumerate(content):
                with open(os.path.join(folder, f"{filename}.{i}"), 'w') as f:
                    f.write(str(item))
        else:
            with open(os.path.join(folder, filename), 'w') as f:
                f.write(content)

    def process_tasks(self, test_tasks: List[str], available_robot_skills: List[dict], objects_ai: str) -> None:
        """
        TaskManager 전체 파이프라인의 main함수
        test_tasks -> 자연어 task 리스트
        available_robot_skills -> 사용가능한 로봇 스킬 리스트
        objects_ai -> 추출한 오브젝트 이름과 무게
        """
        try:
            # 몇개의 task를 처리할건지 확인하는 부분
            print(f"\n[DIAGNOSTIC] Initial Task Count: {len(test_tasks)}")
            
            # objects_ai 정보 저장하기(다른 함수에서 쓰기 위해서)
            self.objects_ai = objects_ai
            
            # 결과 리스트 초기화
            self.decomposed_plan = []
            self.parsed_subtasks = []
            self.precondition_subtasks =[]
            self.subtask_pddl_problems = []
            self.validated_plan = []
            self.subtask_pddl_plans = []
            
            # PDDL 도메인 파일 가져오기
            #(allactionrobot.pddl ->로봇이 어떤 행동을 할수 있고, 어떤 조건이 필요하고, 행동하면 머가 변하는지를 정의한 규칙정리서)
            allaction_domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
            domain_content = self.file_processor.read_file(allaction_domain_path)
            
            # task 단위로 하나씩 돌리기
            for task_idx, (task, robots) in enumerate(zip(test_tasks, available_robot_skills)):
                print(f"\n{'='*50}")
                print(f"Processing Task: {task}: {task_idx + 1}/{len(test_tasks)}")
                print(f"{'='*50}")
                
                # 이번 task 흔적 지우기
                self.clean_all_resources_directories()
                
                # 1. task 분해, decomposed_plan 생성 -> 자연어 task를 여러개의 subtask로 쪼개기 + 필요한 스킬과 오브젝트 선정
                decomposed_plan = self._generate_decomposed_plan(task, domain_content, robots, objects_ai)
                self.decomposed_plan.append(decomposed_plan)
                
                print("✓ Decomposed plan generated")
                print("decomposed plan:\n", decomposed_plan)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                
                # 2. 각각의 서브 테스크 별로 분리 후, 도메인 최적화 정의
                parsed_subtasks = self._decomposed_plan_to_subtasks(decomposed_plan) #분리
                print("✓ Parsed Decomposed Plan generated")
                print("parsed decomposed plan:\n", parsed_subtasks)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                precondition_subtasks = self._generate_precondition_subtasks(parsed_subtasks, domain_content, robots, objects_ai)
                self.precondition_subtasks.append(precondition_subtasks) 
                print("✓ Precondition Decomposed Plan generated")
                print("Precondition Decomposed Plan:\n", precondition_subtasks)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                for item in precondition_subtasks:
                    sid = item.get("subtask_id", -1)
                    title = item.get("subtask_title", "untitled")
                    text = item.get("pre_goal_text", "")

                    safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
                    filename = f"pre_{sid:02d}_{safe_title}.txt"   # 확장자 txt 추천 (PDDL problem이 아니라서)

                    out_path = os.path.join(self.file_processor.precondition_subtasks_path, filename)
                    self.file_processor.write_file(out_path, text)

                #3. 서브테스크에 대한 pddl problem 정의

                subtask_pddl_problems = self._generate_subtask_pddl_problems(precondition_subtasks, domain_content, robots, objects_ai)
                self.subtask_pddl_problems.append(subtask_pddl_problems)

                print("✓ PDDL problems generated")
                print("PDDL problems plan:\n", subtask_pddl_problems)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                for item in subtask_pddl_problems:
                    sid = item["subtask_id"]
                    title = item["subtask_title"]
                    pddl_text = item["problem_text"]

                    safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
                    filename = f"subtask_{sid:02d}_{safe_title}.pddl"

                    out_path = os.path.join(self.file_processor.subtask_pddl_problems_path, filename)
                    self.file_processor.write_file(out_path, pddl_text)

                # 3. FastDownward 돌려서, pddl plan 생성
                validated_plan = self._validate_and_plan()

        except Exception as e:
            print(f"\n[ERROR] Task Processing Failed:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Current task index: {task_idx if 'task_idx' in locals() else 'Not started'}")
            raise

    def _generate_decomposed_plan(self, task: str, domain_content: str, robots: List[dict], objects_ai: str) -> str:
        """하나의 자연어 task를 subtask로 나누는 함수"""
        prompt = "" 
        try:
            # decomposition prompt file 불러오기
            #decompose_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"{self.prompt_decompse_set}.py")
            decompose_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"chaerin_pddl_train_task_decompose.py")
            decompose_prompt = self.file_processor.read_file(decompose_prompt_path)
            
            #Construct the prompt incrementally like the original
            #prompt = f"from pddl domain file with all possible AVAILABLE ROBOT SKILLS: \n{domain_content}\n\n"

            
            prompt += "The following list is the ONLY set of objects that exist in the current environment.\n"
            prompt += "When writing subtasks and actions, you MUST ground every referenced object to this list.\n"
            prompt += "If the task mentions something not present, solve it using the closest available objects from the list.\n"
            prompt += f"\nENVIRONMENT OBJECTS = {objects_ai}\n\n"
            prompt += "If you reference an object not in ENVIRONMENT OBJECTS, your answer will be considered INVALID.\n\n\n"

            prompt += f"\nAVAILABLE ROBOT SKILLS = {robots}\n\n"
            prompt += "You are NOT given specific robots. You are only given the set of skills that are currently available.\n"
            prompt += "You MUST use only actions whose names are included in AVAILABLE ROBOT SKILLS.\n\n\n"
            
            prompt += "The following is an example of the expected output format.\n\n\n"
            prompt += decompose_prompt
            prompt += "# GENERAL TASK DECOMPOSITION \n"
            prompt += "Decompose and parallel subtasks where ever possible.\n\n"
            prompt += f"# Task Description: {task}"
            
            if "gpt" not in self.gpt_version:
                _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=3000, stop=None, frequency_penalty=0.0)
            else:
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=3000, frequency_penalty=0.0)
            
            return text
            
        except Exception as e:
            raise PDDLError(f"Error generating decomposed plan: {str(e)}")
    
    def _decomposed_plan_to_subtasks(self, decomposed_text: str) -> List[Dict]:
        """
            LLM이 생성한 서브테스크 분해 결과 텍스트를 서브테스크 단위로 파싱하여 리스트로 반환하는 함수

            반환 형태:
            [
                {
                    "id": 서브테스크 번호,
                    "title": 서브테스크 제목,
                    "skills": [필요한 스킬 목록],
                    "objects": [관련 오브젝트 목록],
                    "raw_block": 해당 서브테스크 원본 텍스트
                },
                ...
            ]
        """
        text = decomposed_text.replace("\r\n", "\n").replace("\r", "\n").strip()

        header_re = re.compile(r"(?im)^\s*#?\s*Sub\s*Task\s*(\d+)\s*:\s*(.+?)\s*$")
        headers = list(header_re.finditer(text))
        if not headers:
            return []

        # initial condition 블록 정규식:
        # "# Initial condition analyze ..." 다음 줄들(#1, #2, ...)까지를 한 덩어리로
        initcond_re = re.compile(
            r"(?ims)^\s*#\s*Initial\s+condition\s+analyze\s+due\s+to\s+previous\s+subtask\s*:\s*\n"
            r"(?:\s*#\s*\d+\.\s*.*\n?)+"
        )

        subtasks: List[Dict] = []

        for i, h in enumerate(headers):
            start = h.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)

            sub_id = int(h.group(1))
            title = h.group(2).strip()

            # SubTask 본문 블록
            body_block = text[start:end].strip()

            prev_start = headers[i - 1].start() if i > 0 else 0
            prev_region = text[prev_start:start]  # 이전 subtask 시작~현재 subtask 시작 직전

            # prev_region 안에 initial condition이 여러 개 있을 수 있으니, 마지막 매치를 선택
            init_matches = list(initcond_re.finditer(prev_region))
            init_block = init_matches[-1].group(0).strip() if init_matches else ""

            # raw_block 구성: init + subtask body
            raw_block = (init_block + "\n\n" + body_block).strip() if init_block else body_block

            # Skills / Related Objects 추출
            skills = []
            objects = []

            skills_match = re.search(r"(?im)^\s*-?\s*Skills\s+Required\s*:\s*(.+)$", body_block)
            if skills_match:
                skills = [s.strip() for s in skills_match.group(1).split(",") if s.strip()]

            obj_match = re.search(r"(?im)^\s*-?\s*Related\s+Objects?\s*:\s*(.+)$", body_block)
            if obj_match:
                objects = [o.strip() for o in obj_match.group(1).split(",") if o.strip()]

            subtasks.append({
                "id": sub_id,
                "title": title,
                "skills": skills,
                "objects": objects,
                "raw_block": raw_block,
                "initial_conditions": init_block,
            })

        return subtasks

    def extract_domain_header(self, domain_content: str) -> str:
        """
        도메인에서 action 정의 이전까지의 '헤더' 부분만 추출
        (domain 선언, requirements, types, predicates, functions 포함)
        """
        # 첫 action 시작 지점 찾기
        m = re.search(r"(?m)^\s*\(\s*:action\b", domain_content)
        if not m:
            # action이 없다면 그냥 전체 반환
            return domain_content.strip()
        return domain_content[:m.start()].rstrip()

    def extract_action_blocks(self, domain_content: str) -> Dict[str, str]:
        """
        domain_content에서 모든 (:action NAME ... ) 블록을 추출해
        {action_name: action_block_text} 형태로 반환
        """
        # (?s) = DOTALL, 줄바꿈 포함
        # 액션 이름: (:action <NAME>
        # 블록 끝: 다음 (:action ... 또는 도메인 끝의 마지막 ')'
        pattern = re.compile(
            r"(?s)\(\s*:action\s+([A-Za-z0-9_-]+)\b(.*?)\n\s*\)\s*",  
        )

        # 위 패턴은 "액션 끝 괄호"를 정확히 잡기 어려울 수 있어서
        # 더 안전한 방식으로 action 시작 위치들을 모두 찾고, 다음 action 시작 전까지 슬라이스
        starts = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^\s*\(\s*:action\s+([A-Za-z0-9_-]+)\b", domain_content)]
        actions = {}

        if not starts:
            return actions

        for i, (pos, name) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(domain_content)
            block = domain_content[pos:end].rstrip()
            actions[name] = block

        return actions

    def build_subdomain_for_skills(self, domain_content: str, required_skills: List[str]) -> str:
        """
        서브태스크에서 필요한 스킬(action)만 포함한 '축약 도메인 텍스트' 생성
        """
        header = self.extract_domain_header(domain_content)
        action_map = self.extract_action_blocks(domain_content)

        # 중복 제거 + 원래 순서 보존(가능하면)
        seen = set()
        filtered_skills = []
        for s in required_skills:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                filtered_skills.append(s)

        selected_blocks = []
        missing = []
        for skill in filtered_skills:
            if skill in action_map:
                selected_blocks.append(action_map[skill])
            else:
                missing.append(skill)

        # 누락된 스킬이 있다면 주석으로 표시(디버깅에 도움)
        missing_comment = ""
        if missing:
            missing_comment = "\n; WARNING: missing actions in domain: " + ", ".join(missing) + "\n"

        # 도메인 괄호를 닫아야 하므로, header가 "(define ...)"를 이미 열고 있다면 마지막에 ")" 추가 필요
        # 현재 header는 action 이전까지 잘라온 것이므로 보통 아직 domain의 마지막 ')'는 없음.
        subdomain = header.rstrip() + "\n" + missing_comment + "\n\n" + "\n\n".join(selected_blocks) + "\n\n)"
        return subdomain
    
    def _generate_precondition_subtasks(self, parsed_subtasks: List[Dict[str, Any]], domain_content: str, robots: List[dict], objects_ai: str) -> List[Dict[str, Any]]:
        """
        서브태스크 리스트를 입력받아, 각 서브태스크별로 LLM을 호출해 PDDL problem을 위한 precondition과 goal을 추가해 반환해주는 함수

        """
        results: List[Dict[str, Any]] = []

        try:
            # 스킬/오브젝트 목록은 프롬프트 안정성을 위해 정렬
            skills_sorted = sorted(set([s.strip() for s in robots if s and s.strip()]))

            problem_example_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"chaerin_pddl_train_task_pre_pddl_problem.py")
            problem_example_prompt = self.file_processor.read_file(problem_example_prompt_path)

            for st in parsed_subtasks:
                sub_id = st.get("id")
                title = st.get("title", "").strip()
                st_skills = st.get("skills", [])
                st_objects = st.get("objects", [])

                sub_domain = self.build_subdomain_for_skills(domain_content, st_skills)

                prompt = ""
                prompt += "You are a Robot task-to-action expander.\n"
                prompt += "Your job is to EXPAND the given task decomposition into detailed action-level plans using a PDDL-style description.\n"

                prompt += f"CURRENT ENVIRONMENT OBJECT LIST (ground-truth):\n{objects_ai}\n\n"
                prompt += "This is the ONLY set of objects that exist in the current environment.\n"

                prompt += f"OBJECTS SELECTED FROM THE PREVIOUS STEP (for this subtask):\n{st_objects}\n\n"
                prompt += "These are the objects that were judged to be relevant/needed for this subtask.\n"
                prompt += "You should prioritize using these objects when constructing the PDDL problem.\n"
                prompt += "However, you may also use other objects from CURRENT ENVIRONMENT OBJECT LIST if needed.\n\n\n"

                prompt += f"AVAILABLE ROBOT SKILLS (action names):\n{robots}\n\n"
                prompt += "These are the ONLY actions you are allowed to use.\n"

                prompt += f"SKILLS REQUIRED FOR THIS SUBTASK (selected from the previous step):{st_skills}\n"
                prompt += "Your generated PDDL problem must be solvable using ONLY these skills.\n\n\n"

                prompt += f"DOMAIN:\n{sub_domain}\n\n"
                prompt += "There are the domain content containing ONLY the actions you are allowed/required to use.\n"
                prompt += "Use this domain as the sole reference for predicates, action preconditions/effects.\n"

                prompt += "OUTPUT FORMAT CONSTRAINT:\n"
                prompt += "YOUR OUTPUT MUST BE ONLY the following expanded text (no markdown, no explanations)\n\n"

                prompt += "=== EXAMPLE OUTPUT FORMAT START ===\n"
                prompt += problem_example_prompt 
                prompt += "\n=== EXAMPLE OUTPUT FORMAT END ===\n\n"
                prompt += "\n\n=== SUBTASK TO SOLVE ===\n"

                prompt += f"{st}\n\n"  
                
                if "gpt" not in self.gpt_version:
                    _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=2000, stop=["def"], frequency_penalty=0.0)
                else:
                    messages = [{"role": "user", "content": prompt}]
                    _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=2000, frequency_penalty=0.0)
                result = {
                    "subtask_id": sub_id,
                    "subtask_title": title,
                    "skills": st_skills,
                    "objects": st_objects,
                    "pre_goal_text": text,
                    "raw_llm_output": text,
                }
                results.append(result)

            return results

        except Exception as e:
            raise PDDLError(f"Error generating subtask PDDL problems: {str(e)}") from e

    def _generate_subtask_pddl_problems(self, parsed_subtasks: List[Dict[str, Any]], domain_content: str, robots: List[dict], objects_ai: str) -> List[Dict[str, Any]]:
        """
        서브태스크 리스트를 입력받아, 각 서브태스크별로 LLM을 호출해 PDDL problem을 생성하고 반환해주는 함수

        출력:
            List[dict] 형태로 서브태스크별 PDDL problem 결과 반환
            [
              {
                "task_index": 0,
                "subtask_id": 1,
                "subtask_title": "...",
                "skills": [...],
                "objects": [...],
                "problem_name": "...",
                "problem_text": "(define (problem ...))",
                "raw_llm_output": "..."
              },
              ...
            ]
        """
        results: List[Dict[str, Any]] = []

        try:
            # 스킬/오브젝트 목록은 프롬프트 안정성을 위해 정렬
            skills_sorted = sorted(set([s.strip() for s in robots if s and s.strip()]))

            problem_example_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"chaerin_pddl_train_task_pddl_problem.py")
            problem_example_prompt = self.file_processor.read_file(problem_example_prompt_path)

            for st in parsed_subtasks:
                sub_id = st.get("subtask_id")
                title = st.get("subtask_title", "").strip()
                st_skills = st.get("skills", [])
                st_objects = st.get("objects", [])

                sub_domain = self.build_subdomain_for_skills(domain_content, st_skills)

                prompt = ""
                prompt += "You are a PDDL problem generation expert for robot manipulation tasks.\n"
                prompt += "Your job is to generate a valid PDDL *problem* file for the given subtask.\n\n\n"

                prompt += f"CURRENT ENVIRONMENT OBJECT LIST (ground-truth):\n{objects_ai}\n\n"
                prompt += "This is the ONLY set of objects that exist in the current environment.\n"

                prompt += f"OBJECTS SELECTED FROM THE PREVIOUS STEP (for this subtask):\n{st_objects}\n\n"
                prompt += "These are the objects that were judged to be relevant/needed for this subtask.\n"
                prompt += "You should prioritize using these objects when constructing the PDDL problem.\n"
                prompt += "However, you may also use other objects from CURRENT ENVIRONMENT OBJECT LIST if needed.\n\n\n"

                prompt += f"AVAILABLE ROBOT SKILLS (action names):\n{robots}\n\n"
                prompt += "These are the ONLY actions you are allowed to use.\n"

                prompt += f"SKILLS REQUIRED FOR THIS SUBTASK (selected from the previous step):{st_skills}\n"
                prompt += "Your generated PDDL problem must be solvable using ONLY these skills.\n\n\n"

                prompt += f"DOMAIN :\n{domain_content}\n\n"
                prompt += "Use this domain as the sole reference for predicates, action preconditions/effects.\n"

                prompt += "OUTPUT FORMAT CONSTRAINT:\n"
                prompt += "You MUST output ONLY a single complete PDDL problem file.\n"
                prompt += "Do NOT include explanations, markdown, or extra text.\n"
                prompt += "Follow exactly the example format below.\n\n"

                prompt += "=== EXAMPLE OUTPUT FORMAT START ===\n"
                prompt += problem_example_prompt 
                prompt += "\n=== EXAMPLE OUTPUT FORMAT END ===\n\n"
                prompt += "\n\n=== SUBTASK TO SOLVE ===\n"
                prompt += f"{st}\n\n"  

                prompt += "=== WHAT YOU MUST GENERATE ===\n"
                prompt += "Robot starts in the kitchen!\n"
                prompt += "1) (:objects ...) must include ONLY objects that appear in CURRENT ENVIRONMENT OBJECT LIST.\n"
                prompt += "2) (:init ...) must include ALL facts required to make the plan executable.\n"
                prompt += "3) (:goal ...) must represent completion of THIS subtask only.\n\n"

                
                if "gpt" not in self.gpt_version:
                    _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=2000, stop=["def"], frequency_penalty=0.0)
                else:
                    messages = [{"role": "user", "content": prompt}]
                    _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=2000, frequency_penalty=0.0)
                result = {
                    "subtask_id": sub_id,
                    "subtask_title": title,
                    "skills": st_skills,
                    "objects": st_objects,
                    "problem_name": f"subtask_{sub_id}_{re.sub(r'[^a-zA-Z0-9_]+','_', title)[:40]}",
                    "problem_text": text,
                    "raw_llm_output": text,
                }
                results.append(result)

            return results

        except Exception as e:
            raise PDDLError(f"Error generating subtask PDDL problems: {str(e)}") from e

    def _validate_and_plan(self) -> None:
        """pddl problem 파일 검증 후 fastDownward 실행"""
        try:

            self.run_llmvalidator() #검증기

            self.run_planners() #fastDownward
            
        except Exception as e:
            raise PDDLError(f"Error in validation and planning: {str(e)}")

    def run_llmvalidator(self) -> None:
        """Run LLM validation on problem files and store validated .pddl files."""
        try:
            # subtask_pddl_problems_path (LLM이 만든 problem들)
            src_dir = self.file_processor.subtask_pddl_problems_path
            problem_files = [f for f in os.listdir(src_dir) if f.endswith('.pddl')]

            # validated_subtask 폴더 비우기
            self.file_processor.clean_directory(self.file_processor.validated_subtask_path)

            self.validated_plan = []  # 매번 초기화

            for problem_file in problem_files:
                problem_file_full = os.path.join(src_dir, problem_file)

                domain_name = self.file_processor.extract_domain_name(problem_file_full)
                if not domain_name:
                    print(f"[VALIDATOR] No domain specified in {problem_file}")
                    continue

                domain_file = self.file_processor.find_domain_file(domain_name)
                if not domain_file:
                    print(f"[VALIDATOR] No domain file found for domain {domain_name}")
                    continue

                domain_content = self.file_processor.read_file(domain_file)
                problem_content = self.file_processor.read_file(problem_file_full)



                prompt = (
                    "You are a strict PDDL problem validator and repair system for Fast Downward.\n"
                    "The DOMAIN is the single source of truth.\n"
                    "Your job is to REWRITE the PROBLEM so that it is consistent with the DOMAIN "
                    "and solvable when the task intent is achievable.\n\n"

                    f"Domain Description (authoritative):\n{domain_content}\n\n"
                    f"Problem Description (to be repaired):\n{problem_content}\n\n"


                    "Output format must be:\n"
                    "(define (problem <name>)\n"
                    "  (:domain <domain>)\n"
                    "  (:objects ...)\n"
                    "  (:init ...)\n"
                    "  (:goal (and ...))\n"
                    ")\n"
                )
                if "gpt" not in self.gpt_version:
                    _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1400, stop=["def"], frequency_penalty=0.0)
                else:
                    messages = [
                        {"role": "system", "content": "You are a PDDL validator. Output ONLY corrected PDDL."},
                        {"role": "user", "content": prompt}
                    ]
                    _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1400, frequency_penalty=0.0)

                # 정규화(주석/텍스트 제거 + (define부터)
                validated = self.file_processor.normalize_pddl(text)

                # 검증 결과 저장
                self.validated_plan.append(validated)

                out_path = os.path.join(self.file_processor.validated_subtask_path, problem_file)
                self.file_processor.write_file(out_path, validated)
                print(f"[VALIDATED WROTE] {out_path}")

        except Exception as e:
            print(f"Error in run_llmvalidator: {str(e)}")
            raise

    def extract_plan_actions(self, fd_stdout: str) -> list[str]:
        """
        'switchon robot1 faucet (1)'만 추출하는 함수
        """
        plan_actions = []
        action_pattern = re.compile(r'^[a-zA-Z_]+\s+.*\(\d+\)$')

        for line in fd_stdout.splitlines():
            line = line.strip()
            if action_pattern.match(line):
                plan_actions.append(line)

        return plan_actions

    def run_planners(self) -> None:
        """Run PDDL planners on problem files."""
        try:
            planner_path = os.path.join(self.base_path, "downward", "fast-downward.py")
            problem_dir = self.file_processor.validated_subtask_path
            problem_files = [f for f in os.listdir(problem_dir) if f.endswith('.pddl')]  #나중에 검증기 넣을거면 여기 수정해야함
            for problem_file in problem_files:
                try:
                    problem_file_full = os.path.join(problem_dir, problem_file)
                    domain_name = self.file_processor.extract_domain_name(problem_file_full)
                    if not domain_name:
                        print(f"No domain specified in {problem_file}")
                        continue

                    domain_file = self.file_processor.find_domain_file(domain_name)
                    if not domain_file:
                        print(f"No domain file found for domain {domain_name}")
                        continue

                    command = [
                        planner_path,
                        "--alias",
                        "seq-opt-lmcut",
                        domain_file,
                        problem_file_full
                    ]

                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    # 플랜 액션 추출
                    plan_actions = self.extract_plan_actions(result.stdout)
                    self.subtask_pddl_plans.append(plan_actions)

                    base_name = os.path.splitext(problem_file)[0]
                    actions_path = os.path.join(self.file_processor.subtask_pddl_plans_path, f"{base_name}_actions.txt")

                    with open(actions_path, "w") as f:
                        f.write("\n".join(plan_actions))

                    print(f"\n========== FAST-DOWNWARD OUTPUT ({problem_file}) ==========")
                    print(plan_actions)
                    print("===========================================================\n")
                    print(result.stdout)
                    print("===========================================================\n")


                    if result.stderr:
                        print(f"Warnings/Errors for {problem_file}:", result.stderr)
                        
                except subprocess.TimeoutExpired:
                    print(f"Planner timed out for {problem_file}")
                except Exception as e:
                    print(f"Error processing file {problem_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in run_planners: {str(e)}")
            raise

# 커맨드 명령어로 받은 정보 저장하는 함수
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", type=str, help="Path to BDDL file")
    parser.add_argument(
        "--floor-plan", 
        type=int, 
        required=False,  
        help="Required unless --bddl-file is provided"
    )
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument(
        "--gpt-version",
        type=str,
        default="gpt-3.5-turbo",
        choices=['gpt-3.5-turbo', 'gpt-4o', 'gpt-3.5-turbo-16k']
    )
    parser.add_argument(
        "--prompt-decompse-set",
        type=str,
        default="pddl_train_task_decomposesep",
        choices=['pddl_train_task_decompose']
    )
    parser.add_argument(
        "--prompt-allocation-set",
        type=str,
        default="pddl_train_task_allocationsep",
        choices=['pddl_train_task_allocation']
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="final_test",
        choices=['final_test']
    )
    parser.add_argument("--log-results", type=bool, default=True)
    
    args = parser.parse_args()
    
    # 오류 (아무정보도 안 줬을때)
    if not args.bddl_file and args.floor_plan is None:
        parser.error("Either --bddl-file or --floor-plan must be provided")
        
    return args
    
def main():
    """메인 실행 함수"""
    try:
        # 커맨드로 받은 정보들 저장하는 함수 실행(arg.floor_plan = 15, arg.gpt_version = gpt4 같이 저장됨)
        args = parse_arguments()
        
        # task manager 객체 생성 및 초기화 설정
        task_manager = TaskManager(
            base_path=os.getcwd(),
            gpt_version=args.gpt_version,
            api_key_file=args.openai_api_key_file,
            prompt_decompse_set=args.prompt_decompse_set,
            prompt_allocation_set=args.prompt_allocation_set
        )
        
        if args.bddl_file:
            # BDDL파일 실행할때
            print("\nBDDL Data:")
        else:
            # FloorPlan 으로 실행했을때
            test_file = os.path.join("data", args.test_set, f"FloorPlan{args.floor_plan}.json")
            test_tasks, available_robot_skills, gt_test_tasks, trans_cnt_tasks, min_trans_cnt_tasks = \
                task_manager.load_dataset(test_file)
            
            print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
            
            # AI2thor objects 정보 가져오기 
            objects_ai = f"\n\nobjects = {PDDLUtils.get_ai2_thor_objects(args.floor_plan)}"
            
            # objects_ai를 기반으로 tasks 프로세스 시작
            task_manager.process_tasks(test_tasks, available_robot_skills, objects_ai)
            
            # 결과
            if args.log_results:
                for idx, task in enumerate(test_tasks):
                    task_manager.log_results(
                        task=task,
                        idx=idx,
                        available_robots=available_robot_skills,
                        gt_test_tasks=gt_test_tasks,
                        trans_cnt_tasks=trans_cnt_tasks,
                        min_trans_cnt_tasks=min_trans_cnt_tasks,
                        objects_ai=objects_ai
                    )
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(f"Full error: {str(e.__class__.__name__)}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


