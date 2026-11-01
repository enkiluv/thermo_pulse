# -*- coding: utf-8 -*-
"""
================================================================================
benchz.py - High-Difficulty Thermo Pulse Benchmark & Dual-Axis Visualizer (완본)
================================================================================
M1~M9 9가지 디코딩 모드와 고난도 멀티홉 논리(LOGIC) & 고난도 수학(HARD MATH) 
벤치마크 프레임워크 완본 코드입니다.
"""

import os
import sys
import re
import math
import json
import gc
import time
import html
import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from benchz_dataset_v14 import load_dataset_v14

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed
)

# 사전 정의 모델 숏컷
MODEL_PRESETS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct"
}

# =====================================================================
# 1. 고난도 벤치마크 데이터셋 템플릿 (High-Difficulty Logic & Hard Math)
# =====================================================================
LOGIC_TEMPLATES = [
    {
        "category": "Complex 6-Person Seating Arrangement CSP",
        "question": (
            "Six people (A, B, C, D, E, F) sit in a row of six seats numbered 1 to 6 from left to right.\n"
            "1. A sits in seat 2.\n"
            "2. F sits in seat 6.\n"
            "3. C sits immediately between A and D (meaning C is adjacent to both A and D).\n"
            "4. B sits somewhere to the left of E.\n"
            "5. D sits in seat 4.\n"
            "Question: Who is sitting in seat 5? Answer with the person's single letter (A, B, C, D, E, or F)."
        ),
        "answer": "E"
    },
    {
        "category": "Knights and Knaves (Truth-Teller Puzzle)",
        "question": (
            "On an island, Knights always tell the truth and Knaves always lie.\n"
            "Person A says: 'Both Person B and I are Knaves.'\n"
            "Person B says: 'Person A is a Knave.'\n"
            "Question: Is Person A a KNIGHT or a KNAVE? Answer with 'KNIGHT' or 'KNAVE'."
        ),
        "answer": "KNAVE"
    },
    {
        "category": "Multi-hop Propositional Logic Gate Network",
        "question": (
            "System Logic Rules:\n"
            "1. Output Y is TRUE if and only if (Node P is TRUE AND Node Q is FALSE).\n"
            "2. Node P is TRUE if and only if (Input A is 1 OR Input B is 1).\n"
            "3. Node Q is TRUE if and only if (Input C is 1 AND Input D is 1).\n"
            "Facts:\n"
            "- Input A = 1\n"
            "- Input B = 0\n"
            "- Input C = 1\n"
            "- Input D = 0\n"
            "Question: Is Output Y TRUE or FALSE? Answer with 'TRUE' or 'FALSE'."
        ),
        "answer": "TRUE"
    },
    {
        "category": "Extended Temporal Logic Chain with Offsets",
        "question": (
            "Event Timeline Rules:\n"
            "1. Event Alpha occurs 3 hours before Event Beta.\n"
            "2. Event Gamma occurs 5 hours after Event Beta.\n"
            "3. Event Delta occurs 2 hours before Event Gamma.\n"
            "4. Event Epsilon occurs 4 hours before Event Delta.\n"
            "Question: Which event occurs absolute FIRST among Alpha, Beta, Gamma, Delta, and Epsilon? "
            "Answer with the single event name."
        ),
        "answer": "ALPHA"
    },
    {
        "category": "Quantifier Logic with Nested Inclusions",
        "question": (
            "Premises:\n"
            "1. All cyber-analysts are programmers.\n"
            "2. No programmers are careless.\n"
            "3. Some security experts are cyber-analysts.\n"
            "4. All cryptographers are cyber-analysts.\n"
            "Question: Is it definitively true that no cryptographers are careless? "
            "Answer with 'DEFINITELY TRUE', 'DEFINITELY FALSE', or 'UNCERTAIN'."
        ),
        "answer": "DEFINITELY TRUE"
    },
    {
        "category": "Nested Transitive Inequality & Weight Ranking",
        "question": (
            "Premises:\n"
            "1. Box A is strictly heavier than Box B.\n"
            "2. Box C is strictly lighter than Box B.\n"
            "3. Box D is twice as heavy as Box A.\n"
            "4. Box E is strictly lighter than Box C.\n"
            "Question: Which box is the absolute LIGHTEST among A, B, C, D, and E? "
            "Answer with the single box letter."
        ),
        "answer": "E"
    },
    {
        "category": "Multi-step Conditional Disjunction & Contrapositive",
        "question": (
            "Premises:\n"
            "1. Either Server P is active or Server Q is active, but not both.\n"
            "2. If Server P is active, then Alarm R sounds.\n"
            "3. Alarm R sounds if and only if Indicator S lights up.\n"
            "4. Indicator S did NOT light up.\n"
            "Question: Is Server Q active? Answer with 'YES' or 'NO'."
        ),
        "answer": "YES"
    }
]

HARD_MATH_TEMPLATES = [
    {
        "category": "Multi-Stage Work Rate with Efficiency Degradation",
        "question": (
            "A tank has a capacity of 1120 liters and is initially empty.\n"
            "1. Pipe A fills the tank at 120 L/hr and runs alone for 3 hours.\n"
            "2. Pipe B (80 L/hr) is then turned on and works together with Pipe A for 2 hours.\n"
            "3. Then, Drain Pipe C (56 L/hr) is opened, but Pipe A's efficiency degrades by 20%.\n"
            "Question: How many additional hours (after Pipe C opens) will it take to fill the tank completely?\n"
            "Answer with the exact single integer number of hours."
        ),
        "answer": "3"
    },
    {
        "category": "Modular Arithmetic & Chinese Remainder Congruence",
        "question": (
            "Find the smallest positive integer X that satisfies all three congruence conditions:\n"
            "1. X = 3 (mod 5)\n"
            "2. X = 4 (mod 7)\n"
            "3. X = 2 (mod 9)\n"
            "Question: What is the exact value of X? Answer with the single integer number."
        ),
        "answer": "263"
    },
    {
        "category": "Non-Linear Diophantine System & Prime Constraint",
        "question": (
            "A store sells Item A for $50 and Item B for $80. Total revenue from a sale is exactly $1,700.\n"
            "If both items are sold in positive quantities and the total quantity of items (A + B) is a prime number,\n"
            "how many units of Item A were sold?\n"
            "Question: Answer with the single integer quantity of Item A."
        ),
        "answer": "26"
    },
    {
        "category": "Multi-Step Second-Order Recursive Sequence",
        "question": (
            "A sequence is defined recursively by a_1 = 2, a_2 = 5, and a_n = 2*a_{n-1} - a_{n-2} + 3 for n >= 3.\n"
            "Question: What is the exact value of a_8?\n"
            "Answer with the single integer number."
        ),
        "answer": "86"
    },
    {
        "category": "Combinatorial Exclusion with Mutual Antagonism",
        "question": (
            "A committee of 4 people is to be chosen from 5 mathematicians and 4 physicists.\n"
            "However, 1 specific mathematician and 1 specific physicist refuse to serve together on the same committee.\n"
            "Question: How many valid 4-person committees can be formed?\n"
            "Answer with the single integer count."
        ),
        "answer": "105"
    },
    {
        "category": "Multi-Stage Concentration Replacement Cycle",
        "question": (
            "A 100-liter container starts full of 100% pure alcohol.\n"
            "1. First, 20 liters are drawn off and replaced with 20 liters of pure water.\n"
            "2. Second, 25 liters of the mixture are drawn off and replaced with 25 liters of pure water.\n"
            "3. Third, 10 liters of the mixture are drawn off and replaced with 10 liters of pure water.\n"
            "Question: How many liters of pure alcohol remain in the container?\n"
            "Answer with the exact single integer number of liters."
        ),
        "answer": "54"
    },
    {
        "category": "Geometric Border Optimization & Area Offset",
        "question": (
            "A rectangular field's length is 10 meters longer than its width.\n"
            "A paved border of uniform width 2 meters is constructed around the entire outer perimeter of the field.\n"
            "If the area of the paved border alone is exactly 216 square meters,\n"
            "what is the width of the inner field in meters?\n"
            "Question: Answer with the single integer width in meters."
        ),
        "answer": "20"
    }
]

def load_or_create_dataset(domain="logic", json_path="benchmark_v10.json", sample_size=50):
    templates = LOGIC_TEMPLATES if domain == "logic" else HARD_MATH_TEMPLATES
    dataset = []
    for i in range(sample_size):
        tmpl = templates[i % len(templates)]
        dataset.append({"id": i + 1, "question": tmpl["question"], "answer": tmpl["answer"]})
            
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    return dataset

# =====================================================================
# 2. 정밀 정답 파싱 및 Exact Match 평가
# =====================================================================
def parse_extracted_answer(text, domain):
    """Parse only explicit final answer; avoid substring/number-in-reasoning leakage."""
    if not text:
        return ""
    matches = list(re.finditer(r"Final\s+Answer\s*:\s*([^\r\n]+)", text, flags=re.I))
    if not matches:
        return ""
    value = matches[-1].group(1).strip()
    value = re.sub(r"<[^>]*>", "", value).strip()
    value = value.split('.')[0].strip() if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value) else value
    value = value.strip(" \t.,;:!?\"'`*[]()")
    # Expected answer types: signed integer, a single letter, TRUE/FALSE, ON/OFF.
    value = value.upper()
    if re.fullmatch(r"[-+]?\d{1,12}", value):
        return str(int(value))
    if re.fullmatch(r"[A-Z]", value) or value in ("TRUE", "FALSE", "ON", "OFF"):
        return value
    return ""


def evaluate_answer(parsed, target, domain):
    return int(bool(parsed) and str(parsed).strip().upper() == str(target).strip().upper())

def run_mcnemar_test(results_a, results_b):
    b = sum(1 for r_a, r_b in zip(results_a, results_b) if r_a == 1 and r_b == 0)
    c = sum(1 for r_a, r_b in zip(results_a, results_b) if r_a == 0 and r_b == 1)
    a = sum(1 for r_a, r_b in zip(results_a, results_b) if r_a == 1 and r_b == 1)
    d = sum(1 for r_a, r_b in zip(results_a, results_b) if r_a == 0 and r_b == 0)
    table = [[a, b], [c, d]]
    try:
        res = mcnemar(table, exact=True) if (b+c) else None
        p_val = float(res.pvalue) if res is not None else 1.0
    except Exception:
        p_val = 1.0
    return p_val, b, c

# =====================================================================
# 3. M1~M9 디코딩 LogitsProcessor 및 토큰 메타데이터 수집기
# =====================================================================
class ObserveOnlyLogitsProcessor(LogitsProcessor):
    """Measure normalized top-5 entropy for fixed/greedy baselines without altering scores."""
    def __init__(self, temperature):
        self.temperature=temperature
        self.token_logs=[]

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            raw_log_probs=F.log_softmax(scores.float(),dim=-1)
            top=raw_log_probs.topk(5,dim=-1).values
            probs=top.exp();probs=probs/probs.sum(dim=-1,keepdim=True)
            entropy=-(probs*torch.log(probs+1e-12)).sum(dim=-1).mean().item()
            top2=probs[0].topk(2).values
            self.token_logs.append({"step":len(self.token_logs)+1,
                "entropy":float(entropy),"temp":float(self.temperature),
                "margin":float((top2[0]-top2[1]).item()),
                "top1_prob":float(top2[0].item()),"candidate_count":int(scores.shape[-1]),"removed_mass":0.0})
        return scores  # absolutely no decoding intervention

class ThermoPulseLogitsProcessor(LogitsProcessor):
    def __init__(self, mode, T_base=0.50, H_target=1.20, alpha=0.25, beta=0.15, T_min=0.01, T_max=1.20):
        self.mode = mode
        self.T_base = T_base
        self.H_target = H_target
        self.alpha = alpha
        self.beta = beta
        self.T_min = T_min
        self.T_max = T_max
        self.token_logs = []
        self.reset()

    def reset(self):
        self.prev_H = None
        self.current_T = self.T_base
        self.token_logs = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        log_probs = F.log_softmax(scores, dim=-1)
        top_logprobs, _ = torch.topk(log_probs, k=5, dim=-1)
        probs = torch.exp(top_logprobs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        H_t = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item()

        top2_p, _ = torch.topk(probs[0], k=2)
        margin = (top2_p[0] - top2_p[1]).item()
        top1_prob = top2_p[0].item()

        if self.mode == "M6_Thermo_P_Only":
            T_next = self.T_base + self.alpha * (H_t - self.H_target)
        elif self.mode == "M7_Thermo_D_Only":
            delta_H = (H_t - self.prev_H) if self.prev_H is not None else 0.0
            T_next = self.T_base + self.beta * (-delta_H)
        elif self.mode in ["M8_Thermo_Full_PD", "M9_Adaptive_Hybrid"]:
            delta_H = (H_t - self.prev_H) if self.prev_H is not None else 0.0
            T_next = self.T_base + self.alpha * (H_t - self.H_target) + self.beta * (-delta_H)
        elif self.mode == "M4_Entropy_Threshold":
            if H_t < 0.5:
                T_next = 0.05
            elif H_t < 1.5:
                T_next = 0.50
            else:
                T_next = 0.90
        else:
            T_next = self.T_base

        self.current_T = max(self.T_min, min(self.T_max, T_next))
        self.prev_H = H_t

        processed_scores = scores.clone()
        candidate_count = int(scores.shape[-1])
        removed_mass = 0.0
        if self.mode == "M9_Adaptive_Hybrid":
            probs_softmax = F.softmax(processed_scores / self.current_T, dim=-1)
            max_prob = torch.max(probs_softmax, dim=-1, keepdim=True)[0]
            cutoff = max_prob * 0.05
            removed_mask = probs_softmax < cutoff
            candidate_count = int((~removed_mask).sum().item())
            removed_mass = float(probs_softmax[removed_mask].sum().item())
            processed_scores[removed_mask] = float("-inf")

        step = len(self.token_logs) + 1
        self.token_logs.append({
            "step": step, 
            "entropy": float(H_t), 
            "temp": float(self.current_T),
            "margin": float(margin),
            "top1_prob": float(top1_prob),
            "candidate_count": candidate_count,
            "removed_mass": removed_mass,
            "delta_entropy": float(delta_H) if self.mode in ("M7_Thermo_D_Only", "M8_Thermo_Full_PD", "M9_Adaptive_Hybrid") else None
        })

        return processed_scores / self.current_T

class MirostatV2LogitsProcessor(LogitsProcessor):
    def __init__(self, tau=5.0, eta=0.1, T=0.70):
        self.tau = tau
        self.eta = eta
        self.T = T
        self.token_logs = []
        self.reset()

    def reset(self):
        self.mu = 2.0 * self.tau
        self.prev_probs = None
        self.prev_input_len = 0
        self.token_logs = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_len = input_ids.shape[1]
        if self.prev_probs is not None and curr_len > self.prev_input_len:
            last_token_id = input_ids[0, -1].item()
            prob_sampled = self.prev_probs[last_token_id].item()
            surprise = -math.log2(max(prob_sampled, 1e-12))
            self.mu = self.mu - self.eta * (surprise - self.tau)

        scaled_scores = scores / self.T
        probs = F.softmax(scaled_scores, dim=-1)

        log_probs = F.log_softmax(scaled_scores, dim=-1)
        H_t = -torch.sum(probs * log_probs, dim=-1).mean().item()

        top2_p, _ = torch.topk(probs[0], k=2)
        margin = (top2_p[0] - top2_p[1]).item()
        top1_prob = top2_p[0].item()

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        surprises = -torch.log2(torch.clamp(sorted_probs, min=1e-12))

        mask = surprises <= self.mu
        k = max(1, mask.sum().item())

        indices_to_remove = sorted_indices[:, k:]
        processed_scores = scaled_scores.clone()
        processed_scores.scatter_(-1, indices_to_remove, float('-inf'))

        self.prev_probs = probs[0]
        self.prev_input_len = curr_len

        step = len(self.token_logs) + 1
        self.token_logs.append({
            "step": step, 
            "entropy": float(H_t), 
            "temp": float(self.T),
            "margin": float(margin),
            "top1_prob": float(top1_prob), "candidate_count": int(k),
            "mirostat_mu": float(self.mu), "removed_mass": float(sorted_probs[:, k:].sum().item()),
            "surprisal_target_bits": float(self.tau)
        })

        return processed_scores

class FinalAnswerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "Final Answer:" in text:
            after_fa = text.split("Final Answer:")[-1].strip()
            if len(after_fa) > 0 and ("\n" in after_fa or len(after_fa) >= 5):
                return True
        return False

# =====================================================================
# 4. 모델 추론 및 토큰 로깅 제어 함수
# =====================================================================
SYSTEM_INSTRUCTION = (
    "Solve the problem carefully using short intermediate reasoning when needed. "
    "Prioritize a correct answer rather than a guess. Keep reasoning within approximately 2-5 brief sentences. "
    "For the last line write exactly: Final Answer: <ANSWER>. "
    "Do not emit <think> tags or repeat the problem text."
)

def args_max_new_tokens():
    return max(32, int(os.environ.get("BENCH_MAX_NEW_TOKENS", "192")))

def generate_local_response(model, tokenizer, prompt_text, mode, seed=42):
    set_seed(seed)
    
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted_prompt = f"System: {SYSTEM_INSTRUCTION}\nUser: {prompt_text}\nAssistant:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    stopping_criteria = StoppingCriteriaList([FinalAnswerStoppingCriteria(tokenizer, prompt_len)])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    gen_kwargs = {
        "max_new_tokens": args_max_new_tokens(),
        "pad_token_id": pad_token_id,
        "stopping_criteria": stopping_criteria,
        "repetition_penalty": 1.0,
    }

    active_processor = None

    if mode == "M1_Greedy":
        active_processor = ObserveOnlyLogitsProcessor(0.0)
        gen_kwargs["logits_processor"] = LogitsProcessorList([active_processor])
        gen_kwargs["do_sample"] = False
    elif mode == "M2_Fixed_Low":
        active_processor = ObserveOnlyLogitsProcessor(0.20)
        gen_kwargs["logits_processor"] = LogitsProcessorList([active_processor])
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.20
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0
    elif mode == "M3_Fixed_Standard":
        active_processor = ObserveOnlyLogitsProcessor(0.70)
        gen_kwargs["logits_processor"] = LogitsProcessorList([active_processor])
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.70
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0
    elif mode == "M5_Mirostat_v2":
        active_processor = MirostatV2LogitsProcessor(tau=5.0, eta=0.1, T=0.70)
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0
        gen_kwargs["logits_processor"] = LogitsProcessorList([active_processor])
    else:  # M4, M6, M7, M8, M9 (Thermo Pulse 계열)
        active_processor = ThermoPulseLogitsProcessor(mode=mode)
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0
        gen_kwargs["logits_processor"] = LogitsProcessorList([active_processor])

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_count = len(generated_ids)

    token_stream = []
    if active_processor and hasattr(active_processor, "token_logs") and active_processor.token_logs:
        for idx, token_id in enumerate(generated_ids):
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            log_item = active_processor.token_logs[idx] if idx < len(active_processor.token_logs) else {
                "entropy": 0.5, "temp": 0.7, "margin": 0.5, "top1_prob": 0.6
            }
            token_stream.append({
                "token": token_str,
                "step": idx + 1,
                "entropy": log_item["entropy"],
                "temp": log_item["temp"],
                "margin": log_item.get("margin", 0.5),
                "top1_prob": log_item.get("top1_prob", 0.6),
                "candidate_count":log_item.get("candidate_count"),
                "removed_mass":log_item.get("removed_mass"),
                "mirostat_mu":log_item.get("mirostat_mu"),
                "delta_entropy":log_item.get("delta_entropy")
            })
    else:
        temp_val = 0.0 if mode == "M1_Greedy" else (0.20 if mode == "M2_Fixed_Low" else 0.70)
        entropy_val = 0.20 if mode == "M1_Greedy" else (0.40 if mode == "M2_Fixed_Low" else 0.80)
        margin_val = 0.90 if mode == "M1_Greedy" else (0.75 if mode == "M2_Fixed_Low" else 0.40)
        prob_val = 0.95 if mode == "M1_Greedy" else (0.85 if mode == "M2_Fixed_Low" else 0.60)
        
        for idx, token_id in enumerate(generated_ids):
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            token_stream.append({
                "token": token_str,
                "step": idx + 1,
                "entropy": entropy_val,
                "temp": temp_val,
                "margin": margin_val,
                "top1_prob": prob_val
            })

    return generated_text, tokens_count, token_stream

# =====================================================================
# 5. 엔트로피(배경) & 확신도/자유도(글자색/Bold) 2차원 매핑 시각화 엔진
# =====================================================================
def get_entropy_background_color(norm_h: float) -> str:
    val = max(0.0, min(1.0, norm_h))
    if val < 0.4:
        ratio = val / 0.4
        r = int(235 - (235 - 212) * ratio)
        g = int(245 - (245 - 230) * ratio)
        b = int(251 - (251 - 241) * ratio)
    elif val > 0.6:
        ratio = (val - 0.6) / 0.4
        r = int(248 + (250 - 248) * ratio)
        g = int(249 - (249 - 219) * ratio)
        b = int(249 - (249 - 216) * ratio)
    else:
        r, g, b = 248, 249, 249
        
    return f"rgb({r}, {g}, {b})"

def render_experiment_tokens(token_stream: list, pulse_threshold: float = 1.0) -> str:
    if not token_stream:
        return '<span style="color:#64748b;">No token logs recorded for this mode.</span>'

    entropies = [t['entropy'] for t in token_stream]
    margins = [t.get('margin', 0.5) for t in token_stream]
    
    if entropies:
        p_min_h, p_max_h = np.percentile(entropies, 10), np.percentile(entropies, 90)
        if p_max_h == p_min_h: p_max_h += 1e-5
    else:
        p_min_h, p_max_h = 0.0, 1.0

    if margins:
        p_min_m, p_max_m = np.percentile(margins, 10), np.percentile(margins, 90)
        if p_max_m == p_min_m: p_max_m += 1e-5
    else:
        p_min_m, p_max_m = 0.0, 1.0

    html_spans = []
    for item in token_stream:
        tok = html.escape(item['token'])
        step, h_t, t_t = item['step'], item['entropy'], item['temp']
        margin = item.get('margin', 0.5)
        top1_prob = item.get('top1_prob', 0.5)

        norm_h = max(0.0, min(1.0, (h_t - p_min_h) / (p_max_h - p_min_h)))
        bg_color = get_entropy_background_color(norm_h)

        norm_m = max(0.0, min(1.0, (margin - p_min_m) / (p_max_m - p_min_m)))
        
        is_pulse = t_t >= pulse_threshold
        if is_pulse:
            text_color = "#991B1B"
            font_weight = "900"
        elif norm_m > 0.65 or top1_prob > 0.70:
            text_color = "#0F2C59"
            font_weight = "700"
        elif norm_m < 0.25:
            text_color = "#78281F"
            font_weight = "400"
        else:
            text_color = "#2C3E50"
            font_weight = "400"

        pulse_class = "token-pulse" if is_pulse else ""
        badge_html = '<sup class="pulse-badge">⚡</sup>' if is_pulse else ""

        style = f"background-color: {bg_color}; color: {text_color}; font-weight: {font_weight};"
        tooltip = f"Step {step} | H_t: {h_t:.2f} | T_t: {t_t:.2f} | Margin: {margin:.3f} | Top1: {top1_prob:.3f}"
        
        disp_tok = tok.replace("\n", "<br>↵<br>").replace(" ", "&nbsp;")
        html_spans.append(
            f'<span class="token {pulse_class}" style="{style}">{disp_tok}{badge_html}'
            f'<div class="tooltip">{tooltip}</div></span>'
        )

    return "".join(html_spans)

def generate_full_dashboard_html(experiments_results: dict, output_filename: str):
    """Logic 및 Hard Math 결과를 모두 포함하는 대시보드 리포트 생성"""
    cards_html = []
    for card_key, data in experiments_results.items():
        rendered_tokens = render_experiment_tokens(data.get("sample_stream", []), pulse_threshold=1.0)
        card = f"""
        <div class="exp-card" id="card-{card_key.lower().replace(' ', '-').replace('(', '').replace(')', '')}">
            <div class="exp-header">
                <span class="exp-badge">{card_key}</span>
                <div class="exp-title">{card_key} Benchmark Sample Stream</div>
            </div>
            <div class="exp-desc">Acc: {data['mean_acc']}% ± {data['std_acc']}% | Avg Tokens: {data['mean_tokens']}</div>
            <div class="visualizer-box">{rendered_tokens}</div>
        </div>"""
        cards_html.append(card)

    nav_items = "".join([f'<a href="#card-{k.lower().replace(" ", "-").replace("(", "").replace(")", "")}" class="nav-item">{k}</a>' for k in experiments_results.keys()])
    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thermo Pulse Benchmark Visualizer Dashboard (Logic & Math Full)</title>
    <style>
        :root {{
            --bg-color: #F4F6F7; --card-bg: #FFFFFF; --text-color: #2C3E50;
            --text-muted: #5D6D7E; --border-color: #E5E7E9; --accent-navy: #0F2C59; --accent-red: #78281F;
        }}
        body {{ background: var(--bg-color); color: var(--text-color); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 40px 20px; display: flex; justify-content: center; }}
        .container {{ max-width: 1080px; width: 100%; }}
        h1 {{ font-size: 2rem; font-weight: 800; color: #1B4F72; margin-bottom: 8px; }}
        .subtitle {{ color: var(--text-muted); font-size: 1rem; margin-bottom: 24px; }}
        .nav-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 28px; background: var(--card-bg); padding: 12px; border-radius: 12px; border: 1px solid var(--border-color); }}
        .nav-item {{ padding: 6px 14px; background: #EBF5FB; border: 1px solid #AED6F1; color: #1B4F72; text-decoration: none; border-radius: 6px; font-size: 0.85rem; font-weight: 600; transition: all 0.2s; }}
        .nav-item:hover {{ background: #1B4F72; color: #FFF; }}
        .exp-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .exp-badge {{ background: #1B4F72; color: #fff; padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.8rem; font-family: monospace; }}
        .exp-title {{ font-size: 1.2rem; font-weight: 700; display: inline-block; margin-left: 10px; color: #2C3E50; }}
        .exp-desc {{ color: var(--text-muted); font-size: 0.92rem; margin: 10px 0 16px 0; }}
        .visualizer-box {{ background: #FFFFFF; border: 1px solid var(--border-color); padding: 20px; border-radius: 10px; line-height: 2.6; font-size: 1.05rem; }}
        .token {{ display: inline-block; position: relative; padding: 2px 5px; margin: 2px; border-radius: 4px; font-family: ui-monospace, Consolas, monospace; transition: all 0.15s; cursor: pointer; }}
        .token:hover {{ transform: translateY(-2px); z-index: 10; outline: 2px solid #3498DB; }}
        .token-pulse {{ outline: 2px solid #E74C3C; box-shadow: 0 0 8px rgba(231,76,60,0.5); }}
        .pulse-badge {{ color: #E74C3C; font-weight: 900; font-size: 10px; margin-left: 2px; vertical-align: super; }}
        .tooltip {{ position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%); background: #2C3E50; color: #fff; padding: 6px 10px; border-radius: 6px; font-size: 0.75rem; white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 0.2s; border: 1px solid var(--border-color); z-index: 20; font-weight: normal; }}
        .token:hover .tooltip {{ opacity: 1; }}
        .legend {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px 20px; display: flex; gap: 20px; font-size: 0.88em; color: var(--text-muted); flex-wrap: wrap; margin-top: 32px; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-box {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #CCC; }}
    </style>
</head>
<body>
<div class="container">
    <h1>benchz.py :: Thermo Pulse M1 ~ M9 통합 시각화 대시보드</h1>
    <p class="subtitle">논리 추론(LOGIC) & 고난도 수학(HARD MATH) 통합 엔트로피 및 토큰 시각화 리포트</p>
    <div class="nav-bar">{nav_items}</div>
    {"".join(cards_html)}
    <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background:#EBF5FB;"></div> 푸른 배경: 낮은 엔트로피 (확진 구간)</div>
        <div class="legend-item"><div class="legend-box" style="background:#FADBD8;"></div> 붉은 배경: 높은 엔트로피 (고민 구간)</div>
        <div class="legend-item"><span style="color:#0F2C59; font-weight:bold;">ABC</span> 진한 파란 볼드 글자: 높은 확신도 / 독점 선택</div>
        <div class="legend-item"><span style="color:#78281F;">ABC</span> 적갈색 글자: 후보간 경합 / 높은 자유도</div>
        <div class="legend-item"><span style="color:#991B1B; font-weight:900;">ABC⚡</span> 붉은 펄스: Thermo Pulse 발동 Trigger (T_t &ge; 1.0)</div>
    </div>
</div>
</body>
</html>"""
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_html)

# =====================================================================
# 6. 메인 벤치마크 평가 및 결과 출력 파이프라인
# =====================================================================
def evaluate_domain(domain_name, dataset, modes, num_seeds, model, tokenizer):
    print(f"\n" + "="*80)
    print(f"             STARTING EVALUATION FOR DOMAIN: {domain_name.upper()}")
    print("="*80)
    
    results = {}
    for mode in modes:
        print(f"\n[{domain_name.upper()} | Evaluating Mode: {mode}]")
        
        seed_accs = []
        seed_tokens = []
        all_seeds_correct_list = []
        sample_stream_logs = []
        per_seed_correct = []
        per_seed_tokens = []
        per_seed_parsed = []
        diagnostics = {"token_count":0,"entropy_sum":0.0,"high_entropy_tokens":0,"temp_sum":0.0,"candidate_sum":0,"candidate_steps":0,"removed_mass_sum":0.0,"filter_active_steps":0,"mu_sum":0.0,"mu_steps":0}

        for s in range(1, num_seeds + 1):
            seed_val = 42 + s * 10
            correct_list, token_list, parsed_list = [], [], []

            for idx, item in enumerate(dataset):
                gen_text, gen_tokens, stream_logs = generate_local_response(model, tokenizer, item["question"], mode=mode, seed=seed_val)

                if idx == 0 and stream_logs:
                    sample_stream_logs = stream_logs
                for t in stream_logs:
                    diagnostics["token_count"]+=1
                    diagnostics["entropy_sum"]+=float(t["entropy"])
                    diagnostics["temp_sum"]+=float(t["temp"])
                    diagnostics["high_entropy_tokens"]+=int(t["entropy"]>=1.0)
                    if t.get("candidate_count") is not None:
                        diagnostics["candidate_steps"]+=1
                        diagnostics["candidate_sum"]+=int(t["candidate_count"])
                    if t.get("removed_mass") is not None:
                        diagnostics["removed_mass_sum"]+=float(t["removed_mass"])
                        diagnostics["filter_active_steps"]+=int(float(t["removed_mass"])>1e-12)
                    if t.get("mirostat_mu") is not None:
                        diagnostics["mu_sum"]+=float(t["mirostat_mu"])
                        diagnostics["mu_steps"]+=1

                target_ans = item["answer"]
                parsed_ans = parse_extracted_answer(gen_text, domain_name)
                is_correct = evaluate_answer(parsed_ans, target_ans, domain_name)

                correct_list.append(is_correct)
                token_list.append(gen_tokens)
                parsed_list.append(parsed_ans)

                print(f"\rSeed {s}/{num_seeds} | Progress: [{idx + 1}/{len(dataset)} items]", end="", flush=True)

            acc = np.mean(correct_list) * 100.0
            avg_tok = np.mean(token_list)
            seed_accs.append(acc)
            seed_tokens.append(avg_tok)
            all_seeds_correct_list.extend(correct_list)
            per_seed_correct.append(correct_list)
            per_seed_tokens.append(token_list)
            per_seed_parsed.append(parsed_list)

        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        mean_tok = np.mean(seed_tokens)
        std_tok = np.std(seed_tokens)

        arr = np.asarray(per_seed_correct, dtype=int)
        tok_arr = np.asarray(per_seed_tokens, dtype=float)
        consistent_correct = float(np.mean(np.all(arr == 1, axis=0)) * 100.0)
        correct_token_values = tok_arr[arr == 1]
        results[mode] = {
            "mean_acc": round(mean_acc, 2),
            "mechanism_diagnostics": {"mean_top5_entropy":round(diagnostics["entropy_sum"]/max(1,diagnostics["token_count"]),4),"high_entropy_fraction":round(diagnostics["high_entropy_tokens"]/max(1,diagnostics["token_count"]),4),"mean_temperature":round(diagnostics["temp_sum"]/max(1,diagnostics["token_count"]),4),"mean_candidates":round(diagnostics["candidate_sum"]/max(1,diagnostics["candidate_steps"]),2),"mean_removed_probability_mass":round(diagnostics["removed_mass_sum"]/max(1,diagnostics["candidate_steps"]),6),"filter_active_fraction":round(diagnostics["filter_active_steps"]/max(1,diagnostics["candidate_steps"]),4),"mean_mirostat_mu":round(diagnostics["mu_sum"]/max(1,diagnostics["mu_steps"]),3)},
            "std_acc": round(std_acc, 2),
            "mean_tokens": round(mean_tok, 1),
            "std_tokens": round(std_tok, 1),
            "all_seeds_correct": all_seeds_correct_list,
            "sample_stream": sample_stream_logs,
            "seed_by_item_correct": per_seed_correct,
            "seed_by_item_tokens": per_seed_tokens,
            "seed_by_item_parsed": per_seed_parsed,
            "seed_accuracy": [round(float(x), 2) for x in seed_accs],
            "all_seeds_correct_rate": round(consistent_correct, 2),
            "variable_outcome_item_rate": round(float(np.mean((arr.sum(axis=0)>0)&(arr.sum(axis=0)<num_seeds)))*100,2),
            "all_seeds_wrong_rate": round(float(np.mean(np.all(arr==0,axis=0)))*100,2),
            "tokens_per_correct_generation": (round(float(np.mean(correct_token_values)), 2)
                if len(correct_token_values) else None),
            "total_tokens_per_correct_answer": (round(float(tok_arr.sum() / arr.sum()), 2)
                if arr.sum() else None),
            "correct_generations": int(arr.sum()),
            "total_generations": int(arr.size),
            "mechanism_breakdown": {group:{
                "n_items":int(sum(x["mechanism"]==group for x in dataset)),
                "mean_accuracy_pct":round(float(arr[:,[i for i,x in enumerate(dataset) if x["mechanism"]==group]].mean())*100,2),
                "all_seeds_correct_pct":round(float(np.all(arr[:,[i for i,x in enumerate(dataset) if x["mechanism"]==group]]==1,axis=0).mean())*100,2),
                "total_tokens_per_correct_answer":(round(float(tok_arr[:,[i for i,x in enumerate(dataset) if x["mechanism"]==group]].sum()/arr[:,[i for i,x in enumerate(dataset) if x["mechanism"]==group]].sum()),2) if arr[:,[i for i,x in enumerate(dataset) if x["mechanism"]==group]].sum() else None)
                } for group in ("control","boundary","transition","recovery")},
        }
        print(f"\n -> Final Acc: {mean_acc:.2f}% ± {std_acc:.2f}%, Avg Tokens: {mean_tok:.1f} ± {std_tok:.1f}")
        print(f"    All-{num_seeds}-seed correct: {consistent_correct:.2f}%; tokens per correct generation: "
              f"{results[mode]['tokens_per_correct_generation']}; "
              f"total-token cost / correct answer: {results[mode]['total_tokens_per_correct_answer']}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

def _holm_correction(p_values):
    """Holm family-wise error correction, preserving all original keys."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered)-rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def paired_comparisons(logic_results, math_results, datasets, n_boot=5000):
    """Pre-registered primary: M9 vs M3, M9 vs M8, each domain (four contrasts).

    Cluster bootstrap resamples *families*, preserving all questions and five paired
    seeds together. Question-level intervals also reported as exploratory. The twenty
    families are few; confidence intervals may be wide and have limited coverage.
    McNemar uses first fixed seed only, never treats five seeds as independent.
    Family sign-flip exact randomization p is conservative exploratory inference.
    """
    from itertools import product
    out = {}; primary_p = {}
    competitors = ("M1_Greedy", "M2_Fixed_Low", "M3_Fixed_Standard", "M4_Entropy_Threshold",
                   "M5_Mirostat_v2", "M6_Thermo_P_Only", "M7_Thermo_D_Only", "M8_Thermo_Full_PD")
    for domain, results in (("logic", logic_results), ("math", math_results)):
        ds = datasets.get(domain, [])
        if not results or "M9_Adaptive_Hybrid" not in results: continue
        families = np.asarray([r['family'] for r in ds])
        unique_families = np.unique(families)
        a = np.asarray(results["M9_Adaptive_Hybrid"]["seed_by_item_correct"], dtype=float)
        for competitor in competitors:
            if competitor not in results: continue
            b = np.asarray(results[competitor]["seed_by_item_correct"], dtype=float)
            if a.shape != b.shape or a.shape[1] != len(ds):
                raise RuntimeError(f'Misaligned paired arrays: {domain}/{competitor}')
            diff = (a-b).mean(axis=0)
            seed_diff = a[0] - b[0]
            family_diffs = np.array([diff[families == family].mean() for family in unique_families])
            rng = np.random.default_rng(20261027)
            qi = rng.integers(len(diff), size=(n_boot,len(diff)))
            question_ci = np.quantile(diff[qi].mean(axis=1)*100,[.025,.975])
            fi = rng.integers(len(family_diffs), size=(n_boot,len(family_diffs)))
            family_ci = np.quantile(family_diffs[fi].mean(axis=1)*100,[.025,.975])
            # Monte Carlo sign-flip on 20 family-level paired differences.
            # Approximate sensitivity test; not an exact randomization inference for natural model outputs.
            flip_rng=np.random.default_rng(20261028)
            sign_draws=flip_rng.choice((-1,1),size=(50000,len(family_diffs)))
            signed=(sign_draws*family_diffs).mean(axis=1)
            family_p=float((1+np.count_nonzero(np.abs(signed)>=abs(family_diffs.mean())-1e-12))/(len(signed)+1))
            mc_p, mc_b, mc_c = run_mcnemar_test(a[0].astype(int),b[0].astype(int))
            key=f'{domain}_M9_vs_{competitor}'
            is_primary=competitor in ('M3_Fixed_Standard','M8_Thermo_Full_PD')
            out[key]={
                'role':'pre_registered_primary' if is_primary else 'exploratory',
                'mean_accuracy_difference_pp':round(float(diff.mean()*100),3),
                'question_bootstrap_95pct_CI_pp':[round(float(x),3) for x in question_ci],
                'family_cluster_bootstrap_95pct_CI_pp':[round(float(x),3) for x in family_ci],
                'family_sign_flip_p_two_sided':round(family_p,6),
                'family_count':int(len(unique_families)),
                'mcnemar_first_seed_exact':{'p_value':round(mc_p,6),'b':mc_b,'c':mc_c},
                'note':'Five seeds paired per item; 20 families clustered. Sign-flip approximate sensitivity only; nonrandomized model outputs.'
            }
            if is_primary:primary_p[key]=family_p
    adjusted=_holm_correction(primary_p)
    for key,pvalue in adjusted.items():
        out[key]['holm_adjusted_family_sign_flip_p_four_primary_tests']=round(float(pvalue),6)
    return out

def main():
    parser = argparse.ArgumentParser(description="benchz.py - Thermo Pulse & Dual-Axis Visualizer Benchmark")
    parser.add_argument("--model", type=str, default="qwen", help="Model shortcut ('qwen', 'qwen-3b', 'phi3') or HF ID / local path")
    parser.add_argument("--model_path", type=str, default=None, help="Explicit model directory path or HF ID")
    parser.add_argument("--macro", action="store_true", default=True, help="Run both Logic and Hard Math benchmarks (default: True)")
    parser.add_argument("--seeds", "-seeds", "-s", type=int, default=1, help="Number of seeds")
    parser.add_argument("--sample", type=int, default=200, help="Unique questions per domain (V14: 1..200)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--html_output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--start", "--start-mode", type=str, default="M1", help="Start evaluation from a specific mode")
    args = parser.parse_args()

    target_model = args.model_path if args.model_path else MODEL_PRESETS.get(args.model.lower(), args.model)
    is_local = os.path.exists(target_model)
    
    if is_local:
        target_model = os.path.abspath(target_model)
        print(f"[Loading Model] Detected local directory at '{target_model}'...")
    else:
        print(f"[Loading Model] Loading Hugging Face model '{target_model}'...")

    model_tag = "llama" if "llama" in target_model.lower() else "qwen"
    
    output_json = args.output if args.output else f"{model_tag}_results_s{args.sample}.json"
    output_html = args.html_output if args.html_output else f"{model_tag}_report_s{args.sample}.html"

    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True, local_files_only=is_local)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=is_local
    )

    # Fast GPU protocol: reject any CPU/disk layer placement to prevent silent offload slowdown.
    if os.environ.get("BENCH_REQUIRE_FULL_CUDA", "0") == "1":
        devices = set(str(v) for v in getattr(model, "hf_device_map", {}).values())
        if any(x in ("cpu", "disk") for x in devices):
            raise RuntimeError(f"Model offloaded to CPU/disk: {model.hf_device_map}. Use smaller model or reduce memory use.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        print(f"[GPU] Full GPU model placement OK: {devices or {str(model.device)}}", flush=True)

    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.generation_config.repetition_penalty = 1.0

    all_modes = [
        "M1_Greedy", "M2_Fixed_Low", "M3_Fixed_Standard", "M4_Entropy_Threshold",
        "M5_Mirostat_v2", "M6_Thermo_P_Only", "M7_Thermo_D_Only", "M8_Thermo_Full_PD", "M9_Adaptive_Hybrid"
    ]

    start_target = args.start.upper().strip()
    start_idx = 0
    for i, mode in enumerate(all_modes):
        if start_target in mode.upper() or (start_target.isdigit() and int(start_target) == i + 1):
            start_idx = i
            break

    modes = all_modes[start_idx:]
    print(f"\n[Benchmark Config] Model Tag: '{model_tag.upper()}' | Sample: {args.sample} | Seeds: {args.seeds} | Modes: {modes}")

    # 1. 논리(LOGIC) 영역 평가
    logic_dataset = load_dataset_v14("logic", "benchmark_v14_logic_mechanism.json", args.sample)
    logic_res = evaluate_domain("logic", logic_dataset, modes, args.seeds, model, tokenizer)

    # 2. 수학(HARD MATH) 영역 평가
    hard_math_res = {}
    if args.macro:
        hard_math_dataset = load_dataset_v14("math", "benchmark_v14_math_mechanism.json", args.sample)
        hard_math_res = evaluate_domain("math", hard_math_dataset, modes, args.seeds, model, tokenizer)

    # 3. 요약 테이블 2 생성
    table2_data = []
    for mode in modes:
        row = {
            "Configuration": mode,
            "Logic Acc (%)": f"{logic_res[mode]['mean_acc']} ± {logic_res[mode]['std_acc']}",
            "Logic Tokens": f"{logic_res[mode]['mean_tokens']} ± {logic_res[mode]['std_tokens']}",
        }
        if args.macro and mode in hard_math_res:
            row["Math Acc (%)"] = f"{hard_math_res[mode]['mean_acc']} ± {hard_math_res[mode]['std_acc']}"
            row["Math Tokens"] = f"{hard_math_res[mode]['mean_tokens']} ± {hard_math_res[mode]['std_tokens']}"
        table2_data.append(row)

    df_table2 = pd.DataFrame(table2_data)
    print("\n" + "="*80)
    print(f"            TABLE 2: MAIN PERFORMANCE ({model_tag.upper()})            ")
    print("="*80)
    print(df_table2.to_string(index=False))

    mcnemar_summary = {"method":"exact paired McNemar, one fixed seed only; other primary comparisons in paired_comparisons"}

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": target_model,
            "protocol": {"name": "V14_mechanism_200x5_frozen_algorithms", "model": target_model,
                         "new_tokens_cap": args_max_new_tokens(), "sample": args.sample,
                         "seeds": args.seeds, "dataset_seed": 20261031,
                         "note": "200 unique questions/domain, 20 families, four mechanistic categories; five paired seeds. M1-M9 decoding constants unchanged. Categories do not imply M9 superiority. Family sign-flip is sensitivity only."},
            "datasets": {"logic": logic_dataset, "math": hard_math_dataset if args.macro else []},
            "paired_comparisons": paired_comparisons(logic_res, hard_math_res, {"logic":logic_dataset,"math":hard_math_dataset if args.macro else []}),
            "table2": table2_data,
            "mcnemar": mcnemar_summary,
            "logic_raw": logic_res,
            "math_raw": hard_math_res
        }, f, ensure_ascii=False, indent=4)

    # 4. Logic과 Math 시각화 결과를 하나의 HTML 리포트에 통합 바인딩
    visualizer_data = {}
    for mode in modes:
        visualizer_data[f"{mode} (Logic)"] = logic_res[mode]
        if mode in hard_math_res:
            visualizer_data[f"{mode} (Math)"] = hard_math_res[mode]

    generate_full_dashboard_html(visualizer_data, output_html)

    print(f"\n[Saved] Comprehensive Table 2 results written to '{output_json}'.")
    print(f"[Saved] Interactive visualization dashboard saved to '{output_html}'.")

if __name__ == "__main__":
    main()
