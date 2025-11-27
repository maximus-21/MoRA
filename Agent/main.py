import json
import re
import argparse
import time
import os
from tqdm import tqdm
from prompts import*
from utils import*
from agent import*
from openai import OpenAI 
from together import Together
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey


os.environ['TOGETHER_API_KEY'] = ""
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
model = "" # Llama-3-70B, Gemma-2-27B

os.environ['OPENAI_API_KEY'] = ""
llm_model = ""
llm = ChatOpenAI(model=llm_model,)

GRAPH_RAG_DIR = ''
DATASET_DIR = ''
RESULT_DIR = ''

MMLU_Dict = {
    "Modern_Physics": "Modern_Physics",
    "Waves": "Waves",
    "Thermodynamics": "Thermodynamics",
    "Optics": "Optics",
    "Electromagnetism": "Electromagnetism",
    "Mechanics": "Mechanics"
}

SciEval_Dict = {
    "Work and Energy": "Mechanics",
    "Forces and Newton's Laws": "Mechanics",
    "Electrical Energy and Current": "Electromagnetism",
    "Sound": "Waves",
    "Heat": "Thermodynamics",
    "Interference and Diffraction": "Optics",
    "2D Motion": "Mechanics",
    "Subatomic Physics": "Modern Physics",
    "Circular Motion and Gravitation": "Gravitation",
    "Light and Reflection": "Optics",
    "Rotational Motion": "Mechanics",
    "Momentum and Collisions": "Mechanics",
    "Electric Forces and Fields": "Electromagnetism",
    "Waves and Vibrations": "Waves",
    "Fluid Mechanics": "Fluid",
    "Magnetism": "Electromagnetism",
    "Atomic Physics": "Modern Physics",
    "1D Motion": "Mechanics",
    "Thermodynamics": "Thermodynamics",
    "Electromagnetic Induction": "Electromagnetism",
    "Refraction": "Optics",
    "Circuits": "Electromagnetism"
}

PhysicsQA_Dict = {
    "Work Power Energy": "Mechanics",
    "Capacitor": "Electromagnetism",
    "Kinematics 1D": "Mechanics",
    "Alternating Current": "Electromagnetism",
    "Nuclear Physics": "Modern Physics",
    "Simple Harmonic Motion": "Waves",
    "Elasticity": "Elasticity",
    "Communication System": "Semiconductor",
    "Current Electricity": "Electromagnetism",
    "Kinetic Theory of Gases": "Thermodynamics",
    "Rotational Motion": "Mechanics",
    "Waves on String": "Waves",
    "Heat Transfer": "Thermodynamics",
    "Thermal Expansion": "Thermodynamics",
    "Sound Waves": "Waves",
    "Semiconductors": "Semiconductor",
    "Wave Optics": "Optics",
    "Fluid Mechanics": "Fluid",
    "Electrostatics": "Electromagnetism",
    "Magnetism": "Electromagnetism",
    "Friction": "Mechanics",
    "Electromagnetic Waves": "Electromagnetism",
    "Kinematics 2D": "Mechanics",
    "Radioactivity": "Modern Physics",
    "Thermodynamics": "Thermodynamics",
    "Gravitation": "Gravitation",
    "Electromagnetic Induction": "Electromagnetism",
    "Ray Optics": "Optics",
    "Centre of Mass": "Mechanics"
}


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    if 'MMLU' in data_path:
        topic_dict = MMLU_Dict
    elif 'SciEval' in data_path:
        topic_dict = SciEval_Dict
    elif 'PhysicsQA' in data_path:
        topic_dict = PhysicsQA_Dict

    for i in range(len(data)):
        data[i]['INPUT_DIR'] = topic_dict[data[i]['topic']]
    
    return data

def evaluate(dataset_filename, max_steps):

    results = []
    agent = MORA(llm, client, model, graph_llm, token_encoder, text_embedder, max_steps)
    data_path = f'{DATASET_DIR}/{dataset_filename}'
    result_path = f'{RESULT_DIR}/{dataset_filename}'
    data = load_data(data_path)

    for idx, item in enumerate(tqdm(data)):

        result = item

        # print("Id: ", idx, "\n")
        # print("Question: ", item['question'], "\n")
        # print("LLM Response: ", item['response'], "\n")

        INPUT_DIR = f'{GRAPH_RAG_DIR}/{item['INPUT_DIR']}/output/ENTITES/artifacts'
        # LLM Response Refinement
        refined_solution, scratchpad = agent.run(INPUT_DIR, item['question'], item['response'], True)
        result['refined_solution'] = refined_solution
        result['scratchpad'] = scratchpad
        
        results.append(result)

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset_filename', type=str, help="The name of the dataset file to evaluate.")
    parser.add_argument('--max_steps', type=int, default=5, help="The maximum number of iteration steps for refinement.")
    parser.add_argument('--graph_rag_dir', type=str, default='', help="Directory path for GRAPH_RAG data.")
    parser.add_argument('--dataset_dir', type=str, default='', help="Directory path for datasets.")
    parser.add_argument('--result_dir', type=str, default='', help="Directory path for saving results.")

    args = parser.parse_args()

    # Update the directories
    GRAPH_RAG_DIR = args.graph_rag_dir
    DATASET_DIR = args.dataset_dir
    RESULT_DIR = args.result_dir

    # Call the evaluate function with parsed arguments
    evaluate(args.dataset_filename, args.max_steps)




