import jax
import dill
import argparse
from dfax import DFAx, dfax2dfa, dfax2prompt
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler

from learner import Learner
from oracle import Oracle


def dfax2instruct(dfax: DFAx, llm: str, retry_count: int) -> str | None:

    learner = Learner(model_name=llm)
    oracle = Oracle(model_name=llm)

    prompt = dfax2prompt(dfax)
    dfa = dfax2dfa(dfax)

    instruct = learner.reset(prompt)
    check_result, cex = oracle.check(dfa, instruct, retry_count=retry_count, constraint=lambda x: dfa.inputs == x.inputs)

    for _ in range(retry_count):
        if check_result is None:
            return None
        elif check_result is True:
            return instruct
        else:
            wrd_cex, is_pos_cex = cex
            instruct = learner.step(prompt, (wrd_cex, is_pos_cex))
            check_result, cex = oracle.check(dfa, instruct, retry_count=retry_count, constraint=lambda x: dfa.inputs == x.inputs)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DFA to Instruction dataset generator")
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        default=0,
        help="Random seed for PRNG key (default: 0)"
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="?",
        default=-1,
        help="Number of samples to generate (default: -1 -- unbounded)"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        nargs="?",
        default="RAD",
        help="DFA sampler to use (default: RAD, options: R, RA, RAD)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        nargs="?",
        default=5,
        help="Maximum size of DFA to generate (default: 5)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        nargs="?",
        default="qwen3:4b-instruct",
        help="LLM model name (default: qwen3:4b-instruct)"
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        nargs="?",
        default=5,
        help="Number of retries for each sample (default: 5)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed progress output"
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    
    if args.sampler == "R":
        sampler = ReachSampler(max_size=args.max_size)
    elif args.sampler == "RA":
        sampler = ReachAvoidSampler(max_size=args.max_size)
    elif args.sampler == "RAD":
        sampler = RADSampler(max_size=args.max_size, p=None)
    else:
        raise ValueError(f"Invalid sampler choice: {args.sampler}. Valid options are: R, RA, RAD.")

    data = {}
    sample_count = 0
    n = float("inf") if args.n == -1 else args.n

    while sample_count < n:

        if not args.quiet:
            if args.n == -1:
                print(f"Generated {sample_count} samples.", end="\r", flush=True)
            else:
                print(f"Generated {sample_count}/{args.n} samples.", end="\r", flush=True)

        key, subkey = jax.random.split(key)
        dfax = sampler.sample(subkey)

        instruct = dfax2instruct(dfax=dfax, llm=args.llm, retry_count=args.retry_count)
        
        if instruct is None:
            continue

        dfa_data = {dfax: instruct}
        for s in range(dfax.n_states):
            if s == dfax.start:
                continue
            sub_dfax = DFAx(
                start = s,
                transitions = dfax.transitions,
                labels = dfax.labels
            ).minimize()
            sub_instruct = None
            if sub_dfax.n_states == 1:
                if sub_dfax.labels[sub_dfax.start]:
                    sub_instruct = "Success."
                else:
                    sub_instruct = "Failure."
            else:
                sub_instruct = dfax2instruct(dfax=sub_dfax, llm=args.llm, retry_count=args.retry_count)
            if sub_instruct is None:
                break
            dfa_data[sub_dfax] = sub_instruct
        
        if len(dfa_data) == dfax.n_states:
            data.update(dfa_data)
            sample_count += 1
            # Save data using dill
            save_path = f"storage/dataset_seed_{args.seed}_n_{args.n}_sampler_{args.sampler}_max_size_{args.max_size}_llm_{args.llm}_retry_count_{args.retry_count}.pkl"
            with open(save_path, "wb") as f:
                dill.dump(data, f)
    
    if not args.quiet:
        if sample_count > 0:
            print(f"Generated {args.n} samples and saved to: {save_path}")
        else:
            print("No samples generated.")

    # with open(save_path, "rb") as f:
    #     _data = dill.load(f)

    # for dfax in _data:
    #     print(f"DFAx: {dfax}")
    #     print(f"Instruct: {_data[dfax]}")
    #     print("---")

