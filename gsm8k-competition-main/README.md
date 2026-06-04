# GSM8K Competition

[GSM8K](https://github.com/openai/grade-school-math) is a benchmark of ~8.5k grade-school-level math word problems (7.5k train / 1.3k test) released by OpenAI in 2021. Each problem requires 2–8 steps of natural-language reasoning over basic arithmetic. Gold answers are step-by-step with the final number on its own line after `####`. Example:

> **Q:** Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
>
> **A:** Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. `#### 72`

In this competition your `Agent` is evaluated against a private question set in the same format. At test time `Env` draws **K=5 disjoint subsets of 10 questions** each (5 easy + 5 hard, shuffled with a fixed seed). Each subset is sent to your agent as a single batch via `agent.answer(questions: list[str]) -> list[float]` — one numeric answer per question. Each subset has a **60-second timeout**; if a batch fails or times out, no further subsets are sent. Score = number of replies matching the gold answer to within `1e-6`, summed across all delivered subsets. Format errors (replies that are not coercible to float) are reported separately. See `agent_template.py` for the minimal interface and `Agent.py` for a SmolLM3-3B baseline with batched generation and 3-shot prompting. The questions used for the rendering on the plateform are sampled from a different set than the one used for evaluation. 

**HF model cache.** The evaluation platform pre-populates a Hugging Face cache so popular checkpoints don't need to be re-downloaded inside your 60-second batch budget and 100MB upload limit. By default the`AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained` should locate the HF cache and find the available models:

* Qwen/Qwen2.5-0.5B-Instruct
* Qwen/Qwen3-1.7B
* Qwen/Qwen2.5-1.5B-Instruct
* Qwen/Qwen2-1.5B-Instruct
* Qwen/Qwen2.5-Coder-1.5B-Instruct
* deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
* HuggingFaceTB/SmolLM2-1.7B-Instruct
* allenai/OLMo-2-0425-1B-Instruct
* stabilityai/stablelm-2-1_6b-chat
* openbmb/MiniCPM-2B-sft-bf16
* internlm/internlm2_5-1_8b-chat
* tiiuae/Falcon3-1B-Instruct
* LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
* IndexTeam/Index-1.9B-Chat
* TinyLlama/TinyLlama-1.1B-Chat-v1.0
* apple/OpenELM-1_1B-Instruct

Important note : your agent will run in an dedicated container and will not have internet connection : no other API could be called. 


# Ml-arena competition link:  

Enroll with this ml-arena competition ID, and submit your agent on this link:  
[https://ml-arena.com/enroll/05c13e15008e4167b6f91fe67abff6f2](https://ml-arena.com/enroll/05c13e15008e4167b6f91fe67abff6f2)   

# Suggestions for improving your agent:

* Optimizing generation parameters
* Evaluate mathematical expressions with python (ex : 12+6*8 = ?)
* Optimizing the prompt
* Using and combining several generations with high temperature 

# External infos:  

HF LLM course: https://huggingface.co/learn/llm-course/chapter1/1  
HF Agent course: https://huggingface.co/learn/agents-course/unit0/introduction  
