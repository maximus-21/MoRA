QUESTION_BREAKDOWN_PROMPT = """
The following is the given question: {question}

Your task is to generate a detailed breakdown of the question in a tree-like structure. This breakdown should include all the given variables, objectives, and key concepts necessary to solve the question. You do not need to solve the question; just provide the breakdown.

The breakdown should include:
1. Question: question
2. Given Variables: List all variables provided in the question.
3. Objectives: List the main objectives that need to be achieved.
4. Key Concepts: Include the formulas, concepts, and theorems necessary to solve the question.

Example format:
- Question
  - Given Variables:
    - Variable 1
    - Variable 2
    - ...
  - Objectives:
    - Objective 1
    - Objective 2
    - ...
  - Key Concepts:
    - Concept 1: description, formulas, theorems
    - Concept 2: description, formulas, theorems
    - ...

"""

QUESTION_UNDERSTANDING_PROMPT = """
The following is a given question: {question}

Generated Solution: {solution}

Breakdown of the question: {breakdown}

Your task is to verify the following flags for the generated solution:
    1. Does the solution attempts to address the correct objective asked in the question?
    2. Are the correct values of the variables/entities/notations from the question being used in the solution (in the applied formulaes and reasoning)?

Important Notes:
    1. You don't have to verify whether the objective is solved correctly or not.
    2. You don't have to verify whether any reasoning performed, any formulas used, concepts used, or any calculations performed are correct or not.
    3. Focus only on whether the solution addresses the correct objective and uses the correct values of given variables in the applied formulaes.
    4. Also you can't verify for any other variables that solution consider, only verify for the variables given in the question.

Examples:

    1. Objective Flag:
        If the question asks for the average speed of a car, check if the solution is focused on finding the average speed.
    2. Values Flag:
        -If the question provides a distance of 150 miles and a time of 3 hours, ensure these exact values are used in the solution, and not
         some other value such as 120 miles in any step.
        -If the question provides standard variable values, ensure that the solution uses their correct values.
        
Return the flags in a list with 1 for correct and -1 for incorrect, e.g., [1, 1], [1, -1], [-1, 1], [-1, -1].

Output format:
```
Flag: [flag_1, flag_2]
```
"""

CONCEPT_PROMPT = """
The following is a generated solution to the given question {question}:
{solution}

The following is the detailed breakdown of the given question: {breakdown}

Your task is to:
    1. Check the generated solution against the relevant physics concepts and formulaes required to approach the question as in given breakdown.
    2. Verify whether the correct physics concepts and formulas are being used/applied or /not.

Return a concept score determining the stage of incorrect physics concepts/formulae/equation applied, if any. 
The score should be in the range [0, 1], where a lower score indicates an earlier mistake and a higher score indicates a mistake later in the process. A score of 1 means there are no mistakes.

Scoring Guidelines:
  Score Calculation: The score is based on the stage of the given solution where the mistake first occurs.
  - If a mistake occurs at stage n out of N total stages in the solution, the score is calculated as n / N. If n = N i.e mistake is at last step, then n/(N+1).
  - Example: For 5 steps, a mistake at step 4 would result in a score of 0.8.
  - A score of 1 means there are no mistakes.

Important:
    1. The score is intended to verify the correctness of the physics concepts and formulas involved, not the complete correctness of the solution.
    2. Do not verify any mathematical reasoning, computation, algebra or any calculations within the solution. Focus only on whether the correct physics concepts or formulas are being applied or not.
    3. We have a different agent to verify for all the mathematical reasoning, algebra, calculations and computation.

Output Format:
```
Concept Score: [<score>]
```

"""

CALCULATION_PROMPT = """

Given a question and a generated solution, your task is to check for each step of the given solution and verify the mathematical calculations performed in the given solution 
using the code interepeter to generate code for all the compuataion and calculation and verify it.

You have to check all the operations and maths done within the application of the formuales. This includes all arithemtic operation, alegbraic manupilation,
substiutions, application of mathematical procedures (integration, differentiation, etc.), handling of fractions, exponents, and radicals & numerical approximations or rounding.

Import the required libraries to verify the above.

After verifying the given solution, if there is any mistake, determine the step of the solution where the mistake occurs, and return a calculation score.
If there is no mistake in current given solution then return a calculation score of 1.

Scoring Guidelines:
      - The score should be in the range [0, 1], where a lower score indicates a mistake in an earlier step, and a higher score indicates a mistake in a later step.
      - Allowed error tolerance <= ±0.1

Score Calculation:
      - If a mistake occurs at stage n out of N total stages in the current given solution, the score is calculated as n / N. If n = N i.e mistake is at last step, then n/(N+1). Can use code interpreter to calculate the score.
      - Example: For a solution with 5 steps, a mistake at step 4 would result in a score of 0.8 .
                 For a solution with 5 steps, a mistake at step 5 would result in a score of 0.83
      - If there are no mistakes and all the calculation are correct, return a score of 1.
 
Note: 1. You just have to check the mathematical operations mentioned above.
      2. You don't have to perform any kind of reasoning and solve the question. You are not the evaluator of whether the correct concept is used or whether the correct formulae is
         applied etc but you only have to evaluate if the mathematical operations are performed correctly. 

Output Format:
```
Calculation Score: [<score>]
```
"""

USER_PROMPT = """
The following is a generated solution of a given question {question}:
{solution}
"""

REFINE_BREAKDOWN_PROMPT = """
You are tasked with solving a physics problem. Here is the question: {question}
The following is your generated solution: {solution}

In the provided solution, incorrect values of some variables/entities given in the question are being used in some of the formulaes and reasoning. 
Carefully review the question and solution to identify the steps where incorrect values of the given variables are being applied. Once identified, correctly use given values and rework the solution from that step onwards, 
as errors in earlier steps might affect subsequent calculations & reasoning.

Provide a revised solution with the accurate application of the given variables.
"""

REFINE_OBJECTIVE_PROMPT = """
You are tasked with solving a physics problem. Here is the question: {question}
The following is your generated solution: {solution}

In the generated solution, the correct objective of the question is not being addressed. The solutions contains mistakes which leads to
misalignemnt with the objective of the question. Please carefully review the question & understand the objective in detail and regenerate the solution accordingly.
"""

CODE_AGENT_PROMPT = """
You are tasked with correcting the computation for a physics problem.

Question: {question}

Generated Solution: {solution}

Calculation/Computation Score: {score}

The calculation score indicates the stage of the calculation mistake in the given solution. It is represented as n / N. If n = N i.e mistake is at last step, then n/(N+1). Here n is the stage at which the mistake occurred, and N is the total number of steps.

Task:
1. Carefully review the provided solution and identify the step where the incorrect calculation was performed, using the given score as a reference.
2. Generate a Python code to correctly perform the computation and calculation of the stage where the failure occurred. 
3. The code should contain a function "def solve()" which returns a string describing the final computation and calculation result.
4. Make sure all the variables are intialized inside the solve() and it doesn't require any input and also import all the required libraries.
5. Make sure to correctly use variables values with appropiate unit conversions and that variables are properly intialized before they are used.

Ensure that the code includes the following:
- All required arithmetic operations.
- Algebraic manipulations.
- Application of mathematical procedures (e.g., integration, differentiation).
- Value substitution.
- Handling of fractions, exponents, and radicals.
- Numerical approximations or rounding.
- Dimensional analysis.

Use the following format for the code:
```python\n<--Your Code-->\n```
"""

REFINE_CALCULATION_PROMPT = """
You are tasked with solving a physics problem.

Generated Solution: {solution}

Calculation/Computation Score: {score}

The calculation score indicates the stage of the calculation mistake in the given solution. It is represented as n / N. If n = N i.e mistake is at last step, then n/(N+1). Here n is the stage at which the mistake occurred, and N is the total number of steps.

A code agent was activated to generate a Python code to correct the calculation mistake. 

Here's the python Code agent generated : {python_code}
Here's the correct response from the code agent: {code_response}

Task:
1. Identify the step where the calculation mistake occurred based on the given score.
2. Using the correctly calculated response from the code agent, refine the solution.

Provide the refined solution with the corrected calculation.
"""

RETRIEVAL_THOUGHT_PROMPT = """
You are tasked with solving a physics problem. Here is the question: {question}
The following is your generated solution: {solution}
The concept score obtained is: {score}

The score indicates the earliest stage of the incorrect concept/formuale/equation, and is based as n / N. If n = N i.e mistake is at last step, then n/(N+1). Here n is the stage of mistake and N is total number of steps.

The error denotes a failure in applying a correct concept/formulae at a particular step (determined by concept score). You need to:

1. Review the solution & understand the step of failure based on the obtained score.
2. Revisit all the steps in the solution prior to the step of failure, understanding the concept/formuale requirement at the current step of failure.
3. Generate a thought for the relevant formulae/concept required at the current step of failure.

Your retrieval thought should be:
- Simple and sequential
- Identifies the required concept or entities 
- Follows the steps before the point of failure
- Presented as a structured query

Use the following output format:

```
THOUGHT: <your thought>
```

EXAMPLE: What is the kinematic equation to find the final velocity of an object under constant acceleration ?

"""

REFINE_REASONING_PROMPT = """
We are in the process of solving a physics question: {question}
The following is your generated solution: {solution}
The concept score obtained is: {score}

The score indicates the earliest stage of the incorrect concept/formuale/equation, and is based as n / N. If n = N i.e mistake is at last step, then n/(N+1). Here n is the stage of mistake and N is total number of steps.

The error denotes a failure in using a correct concept/formulae at a particular stage based on the score.

Here's the correct retrieved concept/formulae for the failure stage: {observation}

Based on the stage of conceptual mistake, use the correctly retrieved concept/formulae and rework the solution from that step onwards, 
as errors in earlier steps might affect subsequent calculations & reasoning.

Provide the refined solution with corrected reasoning & retrieval.
"""

