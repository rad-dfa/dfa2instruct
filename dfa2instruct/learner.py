from dfa import DFA
from ollama import chat
from typing import List, Tuple

class Learner():
    def __init__(self, model_name="gemma3:1b"):
        self.model_name = model_name
        self.system_prompt = """
Given a Deterministic Finite Automaton (DFA) that represents a task over discrete tokens, your goal is to generate a natural language instruction that precisely and faithfully describes the task presented by the given DFA.

You will be doing this iteratively.
In the first iteration, you will be given a DFA and you should respond with an instruction that describes the task represented by the DFA.
If your answer is correct, then the process terminates.
If your answer is incorrect, then a counterexample that should be either accepted or rejected but is misclassified by your instruction will be generated.
In the consecutive iterations, you will be given the entire history of previous DFAs and your responses along with counterexamples.
You should respond with a new instruction that accurately describes the DFA and correctly classifies all given counterexamples.

Adhere to the following input–output examples.
Do not output anything other than an instruction in quotation marks.
Do not refer to the DFA, states, tokens, or transitions explicitly in your instruction.

#################### Examples #####################

#################### Example 1 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S1
TRANSITIONS:
    S0 -T0-> S1
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 0."

#################### Example 2 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S1
TRANSITIONS:
    S0 -T0-> S1
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 1."

Counterexample:
[0] should be accepted but is rejected by your response.

#### Iteration 1 ####

Prompt:
DFA
STATES: S0, S1
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S1
TRANSITIONS:
    S0 -T0-> S1
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 0."

#################### Example 3 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S2
TRANSITIONS:
    S0 -T8-> S1
    S0 -T9-> S2
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 9 while avoiding token 8."

#################### Example 4 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S2
TRANSITIONS:
    S0 -T8-> S1
    S0 -T9-> S2
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 8."

Counterexample:
[8] should be rejected but is accepted by your response.

#### Iteration 1 ####

Prompt:
DFA
STATES: S0, S1, S2
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S2
TRANSITIONS:
    S0 -T8-> S1
    S0 -T9-> S2
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Avoid token 8."

Counterexample:
[9] should be accepted but is rejected by your response.

#### Iteration 2 ####

Prompt:
DFA
STATES: S0, S1, S2
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S2
TRANSITIONS:
    S0 -T8-> S1
    S0 -T9-> S2
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 9 while avoiding token 8."

#################### Example 5 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T1-> S1
    S1 -T2-> S2
    S2 -T3-> S3
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go tokens 1, 2, and 3, respectively."

#################### Example 6 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T1-> S1
    S1 -T2-> S2
    S2 -T3-> S3
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go tokens 5."

Counterexample:
[1, 2, 3] should be accepted but is rejected by your response.

#### Iteration 1 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T1-> S1
    S1 -T2-> S2
    S2 -T3-> S3
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go tokens 5 while avoiding tokens 2 and 4."

Counterexample:
[1, 2, 3] should be accepted but is rejected by your response.

#### Iteration 2 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T1-> S1
    S1 -T2-> S2
    S2 -T3-> S3
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go token 1."

Counterexample:
[1, 2, 3] should be accepted but is rejected by your response.

#### Iteration 3 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T1-> S1
    S1 -T2-> S2
    S2 -T3-> S3
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go tokens 1, 2, and 3, respectively."

#################### Example 7 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T2-> S3
    S0 -T3-> S2
    S0 -T4-> S1
    S1 -T2-> S2
    S1 -T3-> S2
    S1 -T5-> S0
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 2 while avoiding token 3 at all cost. If you ever reach token 4, then go to token 5 before continuing your mission to token 2."

#################### Example 8 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T2-> S3
    S0 -T3-> S2
    S0 -T4-> S1
    S1 -T2-> S2
    S1 -T3-> S2
    S1 -T5-> S0
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 2 while avoiding token 4."

Counterexample:
[4, 5, 2] should be accepted but is rejected by your response.

#### Iteration 1 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T2-> S3
    S0 -T3-> S2
    S0 -T4-> S1
    S1 -T2-> S2
    S1 -T3-> S2
    S1 -T5-> S0
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to tokens 4, 5, and 2, respectively."

Counterexample:
[2] should be accepted but is rejected by your response.

#### Iteration 2 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T2-> S3
    S0 -T3-> S2
    S0 -T4-> S1
    S1 -T2-> S2
    S1 -T3-> S2
    S1 -T5-> S0
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to token 2 while avoiding token 3. If you go token 4, then you must go to token 5 before continuing your mission to token 2."

#################### Example 9 ####################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2, S3
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S3
TRANSITIONS:
    S0 -T4-> S1
    S0 -T9-> S2
    S2 -T4-> S3
    S2 -T6-> S1
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to tokens 9 while avoiding token 4. Then, go to token 4 while avoiding token 6."

#################### Example 10 ###################

#### Iteration 0 ####

Prompt:
DFA
STATES: S0, S1, S2
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S2
TRANSITIONS:
    S0 -T0-> S1
    S0 -T4-> S2
    S0 -T5-> S1
    S0 -T6-> S2
    S0 -T8-> S2
    S0 -T9-> S1
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to tokens 0, 4, 6, and 8."

Counterexample:
[0] should be rejected but is accepted by your response.

#### Iteration 1 ####

Prompt:
DFA
STATES: S0, S1, S2
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S2
TRANSITIONS:
    S0 -T0-> S1
    S0 -T4-> S2
    S0 -T5-> S1
    S0 -T6-> S2
    S0 -T8-> S2
    S0 -T9-> S1
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

Response:
"Go to tokens 4, 6, and 8 while avoiding tokens 0, 5, and 9."

###################################################

Do not output anything other than an instruction in quotation marks.
Do not refer to the DFA, states, tokens, or transitions explicitly in your instruction.
"""

    def reset(
        self,
        dfa_prompt: str,
    ) -> None:

        self.iteration = 0
        self.prompt = f"#### Iteration {self.iteration} ####\n\nPrompt:\n{dfa_prompt}\n\nResponse:\n"

        response = self.query()

        self.prompt += f"{response}\n\n"

        return response
    
    def step(
        self,
        dfa_prompt: str,
        cex: Tuple[List[int], bool],
    ) -> str:

        self.iteration += 1
        cex, is_pos_cex = cex
        if is_pos_cex:
            self.prompt += f"Counterexample:\n{cex} should be accepted but is rejected by your response.\n\n"
            self.prompt += f"#### Iteration {self.iteration} ####\n\nPrompt:\n{dfa_prompt}\n\nResponse:\n"
        else:
            self.prompt += f"Counterexample:\n{cex} should be rejected but is accepted by your response.\n\n"
            self.prompt += f"#### Iteration {self.iteration} ####\n\nPrompt:\n{dfa_prompt}\n\nResponse:\n"

        response = self.query()

        self.prompt += f"{response}\n\n"

        return response
    
    def query(
        self
    ) -> str:

        result = chat(model=self.model_name, messages=[
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.prompt,
            }
        ])

        response = result.message.content

        return response

