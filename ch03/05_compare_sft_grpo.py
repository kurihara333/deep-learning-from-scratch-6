import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device

device = get_device()
tokenizer = BPETokenizer.load_from('codebot/merge_rules.pkl')

print("SFTモデルを読み込み中...")
sft_model = GPT.load_from('codebot/model_sft.pt', device=device)
print("GRPOモデルを読み込み中...")
grpo_model = GPT.load_from('codebot/model_grpo.pt', device=device)


def format_prompt(user_message):
    return f"### Instruction:\n{user_message}\n\n### Response:\n"


def ask(model, message, max_new_tokens=50, temperature=0):
    prompt = format_prompt(message)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response


# GRPOが実際に強化学習で報酬を与えたタスク(整数の足し算 1〜9)
print("\n" + "=" * 60)
print("算数問題(GRPOが report で直接強化学習したタスク)")
print("=" * 60)
math_questions = ["2+3=", "7+8=", "9+9=", "15+27="]
for q in math_questions:
    sft_ans = ask(sft_model, q, max_new_tokens=20, temperature=0)
    grpo_ans = ask(grpo_model, q, max_new_tokens=20, temperature=0)
    print(f"\n質問: {q}")
    print(f"  SFT : {sft_ans}")
    print(f"  GRPO: {grpo_ans}")


# GRPOでは直接学習していない、一般的な質問
print("\n\n" + "=" * 60)
print("一般的な質問(GRPOでは直接強化学習していない範囲)")
print("=" * 60)
general_questions = ["Hello", "Write a function to reverse a string"]
for q in general_questions:
    sft_ans = ask(sft_model, q, max_new_tokens=200, temperature=1.0)
    grpo_ans = ask(grpo_model, q, max_new_tokens=200, temperature=1.0)
    print(f"\n質問: {q}")
    print(f"  SFT :\n{sft_ans}")
    print(f"  GRPO:\n{grpo_ans}")
