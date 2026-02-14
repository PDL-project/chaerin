<div align="center">

# LLaMAR

**Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments**

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation](https://img.shields.io/badge/docs-coming_soon-red.svg)](https://github.com/nsidn98/LLaMAR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2407.10031-green)](https://arxiv.org/abs/2407.10031)
[![License: MIT](https://img.shields.io/badge/Project-Website-blue)](https://nsidn98.github.io/LLaMAR/)

</div>

A modular vision language model (VLM)-based framework for long-horizon planning for multi-agent robots with partial observability. This is an official implementation of the model described in:

"[Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments](https://arxiv.org/abs/2407.10031)",

Please let us know if anything here is not working as expected, and feel free to create [new issues](https://github.com/nsidn98/LLaMAR/issues) with any questions. Note that we are working to improve the codebase to make it more accessible and easy to use.

## Abstract:

The ability of Language Models (LMs) to understand natural language makes them a powerful tool for parsing human instructions into task plans for autonomous robots. Unlike traditional planning methods that rely on domain-specific knowledge and handcrafted rules, LMs generalize from diverse data and adapt to various tasks with minimal tuning, acting as a compressed knowledge base. However, LMs in their standard form face challenges with long-horizon tasks, particularly in partially observable multi-agent settings. We propose an LM-based Long-Horizon Planner for Multi-Agent Robotics (LLaMAR), a cognitive architecture for planning that achieves state-of-the-art results in long-horizon tasks within partially observable environments. LLaMAR employs a plan-act-correct-verify framework, allowing self-correction from action execution feedback without relying on oracles or simulators. Additionally, we present MAP-THOR, a comprehensive test suite encompassing household tasks of varying complexity within the AI2-THOR environment. Experiments show that LLaMAR achieves a 30% higher success rate than other state-of-the-art LM-based multi-agent planners in MAP-THOR and Search & Rescue tasks.

![image](https://raw.githubusercontent.com/nsidn98/nsidn98.github.io/master/files/Publications_assets/LLaMAR/LLaMAR_arch.png)

**Overview of LLaMAR**: LLaMAR splits the planning process into 4 sub-modules.

- Planner Module - Breaks down high-level language instructions into structured subtasks for execution.
- Actor Module - Selects high-level actions for each agent based on observations and memory.
- Corrector Module - Identifies failures in execution and suggests corrective actions to improve task completion.
- Verifier Module - Confirms subtask completion and updates progress, reducing reliance on external validation.

This iterative Plan-Act-Correct-Verify loop helps with long-horizon multi-agent planning in partially observable environments.

## Usage:

Save your OpenAI API Key in a JSON file named `openai_key.json` with the following format:

```json
{ "my_openai_api_key": "<your_api_key>" }
```

Then execute the python file:

- For MAP-THOR

```bash
python AI2Thor/baselines/llamar/llamar.py --task=0 --floorplan=0
```

This should run an episode on the `task=0` (Put the bread, lettuce and tomato in the fridge) on `floorplan=0` (FloorPlan1)

You can refer to `configs/config_type1.json` for the mapping on tasks and floorplans.

OR

- For Search & Rescue

```bash
python SAR/baselines/llamar.py --scene=1 --name='llamar_SAR' --agents=2 --seed=0
```

This should run an episode in a scene with one fire with 2 initially lighted cells, the other fire has 1 initially lighted location. You can refer `SAR/Scenes` for more description on the scenes.

Note: The experiments in the paper were obtained with `gpt-4-vision-preview` which has been [deprecated](https://platform.openai.com/docs/deprecations). OpenAI [suggests](https://platform.openai.com/docs/guides/vision?lang=node) using `gpt-4o-mini`. Some of the numbers from the paper might not match.

Warning: Running all the experiments from the paper cost us tokens worth around $3000 (due to the pricing structure of GPT-4 at the time of the experiments).

## Repo Structure:

```
├── README.md               # Project documentation
├── README_mapthor.md       # MAPTHOR documentation
├── requirements.txt        # Dependencies and package requirements
├── AI2Thor/                # everything related to the AI2THOR experiments
├── SAR/                    # everything related to the SAR experiments
├── configs/                # Configuration files for different MAPTHOR tasks
├── plots/                  # Code used to generate plots/tables in the paper
├── thortils/               # Some utility functions for AI2THOR
├── vlms/                   # Open source VLMs
├── init_maker/             # Code to create new scene initialisations
├── results/                # Some logs/outputs from the VLMs
└── .gitignore              # Files to ignore in version control
```

## Dependencies:

Refer `requirements.txt` for other libraries

- `pip install openai==0.27.4`
- `pip install ai2thor==5.0.0`
- `sentence-transformers==2.3.1`
- `transformers==4.38.0`
- `pip install open3d==0.16.1`
- `pip install opencv-python==4.7.0.72`

## Questions/Requests

Please file an issue if you have any questions or requests about the code or the paper.

## Citation

If you found this codebase useful in your research, please consider citing

```bibtex
@inproceedings{llamar,
  title={Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments},
  author={Nayak, Siddharth and Orozco, Adelmo Morrison and Ten Have, Marina and Zhang, Jackson and Thirumalai, Vittal and Chen, Darren and Kapoor, Aditya and Robinson, Eric and Gopalakrishnan, Karthik and Harrison, James and Ichter, Brian and Mahajan, Anuj and Balakrishnan Hamsa},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

```bibtex
@inproceedings{
nayak2024mapthor,
title={{MAP}-{THOR}: Benchmarking Long-Horizon Multi-Agent Planning Frameworks in Partially Observable Environments},
author={Siddharth Nayak and Adelmo Morrison Orozco and Marina Ten Have and Vittal Thirumalai and Jackson Zhang and Darren Chen and Aditya Kapoor and Eric Robinson and Karthik Gopalakrishnan and Brian Ichter and James Harrison and Anuj Mahajan and Hamsa Balakrishnan},
booktitle={Multi-modal Foundation Model meets Embodied AI Workshop @ ICML2024},
year={2024},
url={https://openreview.net/forum?id=ZygZN5egzy}
}
```

## Contributing

We would love to include more scenes in MAP-THOR and would be happy to accept PRs.

## License

MIT License
