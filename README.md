# MoRA
Code Repository for: [Improving Physics Reasoning in Large Language Models Using Mixture of Refinement Agents](https://arxiv.org/abs/2412.00821) [AAAI 2026 TrustAgent]

PhysicsQA: [HuggingFace](https://huggingface.co/datasets/maximus-21/PhysicsQA)

## Requirements

The code is written in Python 3.8. Before running, you need to first install the required packages by typing following commands (Using a virtual environment is recommended):

```
conda create --name mora python==3.8
conda activate mora
pip3 install -r requirements.txt
```

## Datasets
You can access the datasets in `MoRA/Dataset`

**Datasets:**

1. `MMLU_College_Physics.json`
2. `MMLU_High_School_Physics.json`
3. `PhysicsQA.json`
4. `SciEval_Static_Physics.json`

## Agent

MoRA implementation can be found in `MoRA/Agent`

**Python Files:**

1. `agent.py`: Contains the `MoRA` agent class
2. `prompts.py`: Contains all the error identification & refinement prompts
3. `utils.py`: Contains utilities functions for using GraphRAG
4. `main.py`: Contains `evaluate()` function for running MoRA on datasets

## Knowledge Base

Download the `GRAPH_RAG` folder which contains Topic wise KBs from here: [Google Drive](https://drive.google.com/file/d/1reSQgvrqGwh_lNEXLbJlbaRCHIYnmDLd/view)

Replace empty  `MoRA/GRAPH_RAG` folder with the downloaded one.

Using GraphRAG, local search is performed on these KBs to obtain conceptual contexts.

## Experiments

### Run MoRA

**Steps:**

1. Add your `TOGETHER_API_KEY` &  `OPENAI_API_KEY` in `main.py`

2. Add your `GRAPHRAG_API_KEY` in `utils.py`, it's same as your `OPENAI_API_KEY`

3. Add `model` in `main.py` as base LLM:

   1. `Llama-3-70B`: "meta-llama/Llama-3-70b-chat-hf"
   2. `Gemma-2-27B`: " google/gemma-2-27b-it"

   You can select any other LLM as well, refer to for model path: [TogetherAI](https://docs.together.ai/docs/chat-models)

4. Add `llm_model` in `main.py` as error identifier, for our experiments we used: `gpt-4o` 

5.  Run `main.py` with following args:

   1. `dataset_filename`: The name of the dataset file to test, example `PhysicsQA.json` , make sure dataset contains `response` for each questions that needs to be refined. Run `CoT` inference first on the dataset (see the next section) and use that filename and path.
   2. `max_steps`: The maximum number of iteration steps for refinement
   3. `graph_rag_dir`: Directory path for GRAPH_RAG data
   4. `dataset_dir`: Directory path for datasets, should include `response` for each question that needs to be refined. 
   5. `result_dir`: Directory path for saving results

   Example: 

   ```
   python main.py PhysicsQA.json --max_steps 5 --graph_rag_dir 'MoRA/GRAPH_RAG' --dataset_dir 'MoRA/Dataset' --result_dir 'MoRA/Results'
   ```

   

