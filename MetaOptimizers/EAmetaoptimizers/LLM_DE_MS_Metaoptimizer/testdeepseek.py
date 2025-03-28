from openai import OpenAI
import re
from pathlib import Path
def process_string(decs,fits,cv,fn,curiter,history):
    client = OpenAI(
        api_key = "sk-5282cc59c61242d0aaf7f0a0d4ffcc30",
        base_url ="https://api.deepseek.com",
    )
    userprompt=f"""
    **Task**: Design a novel and effective mutation equality for this optimization problem according to the relationship among the decision vectors {decs}, fitness vectors {fits}, and constraints violation {cv} for this population to generate offspring.
    
    **Requirement**:
    1. Output the update equality or rules using mathematics equality or pseudo code in latex format.
    2. Output a correct and complete MATLAB code file named 'updateFunc{curiter}.m', the function need to define as follows whose name is 'updateFunc' plus the current iteration int number:
       function [offspring] = updateFunc{curiter}(popdecs, popfits, cons)
    3. The update function prefers vectorization operations and avoiding using toolbox functions.
    4. Check the generated MATLAB code carefully which must have no endless loop in the function.
    
    **History Feedback**:
    {history}
    
    **Output Farmat**:
    ```latex
    % update rule
    ```

    ```matlab
    % MATLAB Code
    ```
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"You are an operation and optimization expert and designing novel differential evolution algorithms for problem CEC2017 F{fn}."},
                {"role": "user", "content": userprompt},
            ],
            stream=False,
            temperature=0.3
        )
        
        if response is None:
            raise ValueError("Response from the API is None. Please check the API call or network connection.")
        
        response_content = response.choices[0].message.content
        print("Response content:", response_content)

    except Exception as e:
        responsecontent = ""
        
    code = extract_matlab_code(responsecontent)
    save_generation(float(fn), float(curiter), responsecontent, userprompt)
    return code


def extract_matlab_code(text):
    match = re.search(r'```matlab\n(.*?)\n```', text, re.DOTALL)
    return match.group(1) if match else ""

# def load_history(functionnum, current_iter):
#     offset = (functionnum - 1) * 10 
#     base_step = 280
#     num_bases = 20

#     if (
#         (current_iter - offset) % base_step == 0 
#         and 0 <= (current_iter - offset) // base_step < num_bases
#     ) or not history:
#         return "No history"
#     last = history[-1]
#     analysis = [
#         f"Last Result: MinValue={last['metrics']['MinValue']:.3f}, FeasibleRate={last['metrics']['FeasibleRate']:.4f}",
#         f"Runtime: {last['metrics']['time']:.1f}s",
#         f"Error: {last['error'] or 'no'}"
#     ]
#     return "\n".join(analysis)
    

def save_generation(functionnum, curiter, response, prompt):
    gen_dir = Path('Data/generations')
    gen_dir.mkdir(exist_ok=True)
    
    (gen_dir / f'cec2017F{functionnum}_iter{(curiter//280)*10+curiter%280-(functionnum-1)*10}_response.txt').write_text(response,encoding="utf-8",errors = 'ignore')
    (gen_dir / f'cec2017F{functionnum}_iter{(curiter//280)*10+curiter%280-(functionnum-1)*10}_prompt.txt').write_text(prompt,encoding="utf-8",errors = 'ignore')

# a="[1,2,3;4,5,6]"
# b="[1,2]"
# c="[0,1]"
# d=1
# e=1
