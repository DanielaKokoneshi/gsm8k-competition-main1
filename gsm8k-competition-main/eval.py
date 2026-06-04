"""Minimal local evaluator: load Agent from agent_llm, run Env.evaluate, print."""

from env import Env, wrap
#from agent_llm import Agent
from to_import.agent  import Agent


def main():
    env = Env()
    agent = wrap(Agent())
    result = env.evaluate([agent], [{"agent_index": 0}])
    print(result["agent_results"][0])


if __name__ == "__main__":
    main()
