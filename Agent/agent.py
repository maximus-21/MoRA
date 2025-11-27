import re
import time
from openai import OpenAI 
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from prompts import*
from utils import*


breakdown_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Physics Assistant Agent."),
    ("human", QUESTION_BREAKDOWN_PROMPT),
])

question_understanding_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Physics Assistant Agent."),
    ("human", QUESTION_UNDERSTANDING_PROMPT),
])

concept_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Physics Assistant Agent."),
    ("human", CONCEPT_PROMPT),
])

calculation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calculation verifying Agent."),
    ("human", CALCULATION_PROMPT),
])

class MORA:

    def __init__(self, llm, client, model, graph_llm, token_encoder, text_embedder, max_steps = 10):
        self.max_steps = max_steps
        self.client = client
        self.llm = llm
        self.graph_llm = graph_llm
        self.model = model
        self.client_gpt = OpenAI()
        self.assistant = self.client_gpt.beta.assistants.create(instructions=CALCULATION_PROMPT, model="gpt-4o", tools=[{"type": "code_interpreter"}])
        self.thread = self.client_gpt.beta.threads.create()
        self.token_encoder = token_encoder
        self.text_embedder = text_embedder
        self.__reset_agent()

    def run(self, INPUT_DIR, question, solution, reset):
        if reset:
            self.__reset_agent()

        self.question = question
        self.solution = solution
        self.breakdown = self.llm.invoke(breakdown_prompt.format_messages(question=self.question)).content
        self.INPUT_DIR = INPUT_DIR
        self.finished = False
        self.scratch_pad = ""
    
        # Iterative Refinement
        while not self.finished and self.step_n < self.max_steps:
            self.step()
            if not self.finished:
                self.step_n += 1
            self.scratch_pad += f"\n---------------------STEP {self.step_n}-----------------------\n"
            # print("----------STEP ",self.step_n, " ---------------\n")
            self.scratch_pad += f"{self.solution}\n---------------------------------------------------\n" 
            # print(self.solution, "\n------------------------------------------------\n")
           

        if self.step_n >= self.max_steps:
            print("Max steps limit reached")

        return self.solution, self.scratch_pad
    
    def step(self):
        self.solution = self.refinement(self.solution)
    
    def gpt_router(self, solution):
        ##-------------------Problem Comprehension Flags-------------------
        question_understanding_flags = self.llm.invoke(question_understanding_prompt.format_messages(question = self.question, solution = solution, breakdown = self.breakdown)).content
        # print("\nQuestion Understanding Flag: ", question_understanding_flags, "\n")
        question_understanding_flags = eval(self.parse_gpt_response(question_understanding_flags))
        self.scratch_pad += f"\nQuestion Understanding Flag: {question_understanding_flags}\n"

        ##-------------------Concept Verification Score--------------------
        concept_score = self.llm.invoke(concept_prompt.format_messages(question = self.question, solution = solution, breakdown = self.breakdown)).content
        # print("\nConcept Score: ", concept_score, "\n")
        concept_score = eval(self.parse_gpt_response(concept_score))
        self.scratch_pad += f"\nConcept Score: {concept_score}\n"
   

        ##------------------Computation Verification Score------------------
        cal_score = self.cal_verification(question = self.question, solution = solution)
        # print("\nCalculation Score: ", cal_score, "\n")
        cal_score = eval(self.parse_gpt_response(cal_score))
        self.scratch_pad += f"\nCalculation Score: {cal_score}\n\n"

        return question_understanding_flags, concept_score, cal_score
    
    def refinement(self, solution):
        # Error identification with GPT-4o
        question_understanding_flags, concept_score, cal_score = self.gpt_router(solution)

        if isinstance(concept_score, list):
            concept_score = concept_score[0]

        if isinstance(cal_score, list):
            cal_score = cal_score[0]

        # Prioritized Routing
        if question_understanding_flags[0] != 1:
            refined_solution = self.llama_response(REFINE_OBJECTIVE_PROMPT.format(question = self.question, solution = solution))
            return refined_solution
        elif question_understanding_flags[1] != 1:
            refined_solution = self.llama_response(REFINE_BREAKDOWN_PROMPT.format(question = self.question, solution = solution))
            return refined_solution
        elif concept_score < 0.99:
            refined_solution = self.concept_agent(solution, concept_score)
            return refined_solution
        elif cal_score < 0.99:
            refined_solution = self.cal_agent(solution, cal_score)
            return refined_solution
        
        self.finished = True

        return solution ## No refinement
    
    def concept_agent(self, solution, concept_score):
        # Retrieval Thought Generation
        thought = self.parse_llama_response(self.llama_response(RETRIEVAL_THOUGHT_PROMPT.format(question = self.question, solution = solution, score = concept_score)))
        self.scratch_pad += f"THOUGHT: {thought}\n"
        # print("THOUGHT: ", thought, "\n")

        # Concept Context Retrieval using GraphRAG
        observation = self.local_search(thought)
        self.scratch_pad += f"OBSERVATION: {observation}\n"
        # print("OBSERVATION: ", observation, "\n")

        # Solution Refinement using Observation
        refined_ans = self.llama_response(REFINE_REASONING_PROMPT.format(question = self.question, solution = solution, score = concept_score, observation = observation))
        return refined_ans

    def cal_agent(self, solution, cal_score):
        # Code Generation for correct computations
        cal_code = self.parse_llama_response(self.llama_response(CODE_AGENT_PROMPT.format(question=self.question, solution=solution, score=cal_score)))
        self.scratch_pad += f"------------------------Python Code---------------------\n{cal_code}\n------------------------------------------------------\n"
        # print("---------------Python Code--------------\n", cal_code, "\n----------------------------------------------\n")
        local_context = {}

        # print("Code Response:", end=" ")
        # Code Execution for code response
        exec(cal_code, globals(), local_context)
        code_response = local_context['solve']()
        self.scratch_pad += f"\nCode Response: {code_response}\n"

        # Solution Refinement using code response
        refined_ans = self.llama_response(REFINE_CALCULATION_PROMPT.format(question=self.question, solution=solution, score=cal_score, python_code = cal_code ,code_response=code_response))
        return refined_ans
        
    def llama_response(self, llm_input):
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": llm_input}],
        )
        return response.choices[0].message.content
    
    def cal_verification(self, question, solution):
        # Calculation Verifcation using OpenAI Code Interpreter

        message = self.client_gpt.beta.threads.messages.create(
        thread_id=self.thread.id,
        role="user",
        content= USER_PROMPT.format(question = question, solution = solution))

        run = self.client_gpt.beta.threads.runs.create(thread_id=self.thread.id, assistant_id=self.assistant.id)

        timeout = 180
        interval_time = 5
        time_taken = 0
        while time_taken < timeout:
            run = self.client_gpt.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=run.id)

            if run.status == 'completed':
                messages = self.client_gpt.beta.threads.messages.list(thread_id=self.thread.id)
                return messages.data[0].content[0].text.value
            else:
                time.sleep(interval_time)
                time_taken += interval_time
                
        print("TimeoutError")
        return None
    
    def local_search(self, query):
        # Local Search on KB using GraphRAG

        entities, entity_df, description_embedding_store = get_entities(self.INPUT_DIR)
        relationships = get_relationships(self.INPUT_DIR)
        reports = get_reports(self.INPUT_DIR, entity_df)
        text_units = get_text_units(self.INPUT_DIR)

        context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
        )

        search_engine = LocalSearch(
        llm = self.graph_llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

        result = search_engine.search(query)
        return result.response

    def parse_llama_response(self, response):
        if 'THOUGHT' in response:
            expression = f'THOUGHT:'
            pattern = re.compile(f"{expression}\s*(.*)")
            matches = pattern.findall(response)
            parsed_response = matches[-1] if matches else None
        elif 'python' in response:
            pattern = r'```python\n(.*?)\n```'
            matches = re.findall(pattern, response, re.DOTALL)
            parsed_response = matches[-1] if matches else None
        if parsed_response is None:
            print("Incorrect Response:\n", response)

        return parsed_response 
        
    def parse_gpt_response(self, response):
        if "Flag" in response:
            expression = f'Flag:'
            pattern = re.compile(f"{expression}\s*(.*)")
            matches = pattern.findall(response)
            parsed_response = matches[-1] if matches else None
        elif "Calculation Score" in response:
            expression = f'Calculation Score:'
            pattern = re.compile(f"{expression}\s*(.*)")
            matches = pattern.findall(response)
            parsed_response = matches[-1] if matches else None
        elif "Concept Score" in response:
            expression = f'Concept Score:'
            pattern = re.compile(f"{expression}\s*(.*)")
            matches = pattern.findall(response)
            parsed_response = matches[-1] if matches else None
        else:
             parsed_response = None
             print("Incorrect Response:\n", response)
        
        return parsed_response
    
    def __reset_agent(self) -> None:
        self.step_n = 0
        self.finished = False