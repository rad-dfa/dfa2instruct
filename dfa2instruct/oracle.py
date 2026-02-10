from dfa import DFA
from ollama import chat
from dfax import prompt2dfax, dfax2dfa
from typing import Callable, List, Tuple

class Oracle():
    def __init__(self, model_name="gemma3:1b"):
        self.model_name = model_name
        self.system_prompt = """
Given a natural language instruction that explains a task over discrete tokens, your goal is to generate a Deterministic Finite Automaton (DFA) of the given instruction.

You will use a specific format to represent the DFA, which includes the states, tokens, initial state, accepting states, and transitions.
The format is as follows:
    DFA
    STATES: <list of states separated by commas>
    TOKENS: <list of tokens separated by commas>
    INIT_STATE: <initial state>
    ACCEPTING_STATES: <list of accepting states separated by commas>
    TRANSITIONS:
        S -T-> S' <denotes a transition from state S to state S' on token T>
        For every state Si and token Tj not listed above, the transition goes back to the same state:
            Si -Tj-> Si for all other state-token pairs. <omited transitions are self-loops>

Adhere to the following input–output examples.
Output in the given format only.

Examples:

#################### Example 1 ####################

Prompt:
"Go to token 0."

Response:
DFA
STATES: S0, S1
TOKENS: T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
INIT_STATE: 0
ACCEPTING_STATES: S1
TRANSITIONS:
    S0 -T0-> S1
    For every state Si and token Tj not listed above, the transition goes back to the same state:
        Si -Tj-> Si for all other state-token pairs.

#################### Example 2 ####################

Prompt:
"Go to token 9 while avoiding token 8."

Response:
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

#################### Example 3 ####################

Prompt:
"Go tokens 1, 2, and 3, respectively."

Response:
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

#################### Example 4 ####################

Prompt:
"Go to token 2 while avoiding token 3 at all cost. If you ever reach token 4, then go to token 5 before continuing your mission to token 2."

Response:
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

#################### Example 5 ####################

Prompt:
"Go to tokens 4, 6, and 8 while avoiding tokens 0, 5, and 9."

Response:
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

#################### Example 6 ####################

Prompt:
"Go to tokens 9 while avoiding token 4. Then, go to token 4 while avoiding token 6."

Response:
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

###################################################

Output in the given format only.
"""

    def parse(
        self,
        dfa_prompt: str,
    ) -> DFA | None:
        try:
            return dfax2dfa(prompt2dfax(dfa_prompt))
        except:
            return None

    def check(
        self,
        dfa: DFA,
        instruct: str,
        retry_count: int = 10,
        constraint: Callable = lambda dfa: True
    ) -> Tuple[bool | None, Tuple[List, bool] | None]:

        user_prompt = f"Prompt:\n{instruct}\n\nResponse:"
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        for _ in range(retry_count):
            result = chat(model=self.model_name, messages=messages)
            response = result.message.content
            dfa_hat = self.parse(response)
            if isinstance(dfa_hat, DFA) and constraint(dfa_hat):
                wrd_cex = (dfa ^ dfa_hat).find_word()
                if wrd_cex is not None:
                    is_pos_cex = dfa.label(wrd_cex, start=dfa.start)
                    return False, (wrd_cex, is_pos_cex)
                else:
                    return True, None
        return None, None

