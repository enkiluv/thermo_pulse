"""Thermo Pulse M9: external entropy-based P/D sampling over Qwen3.5 logits.

The M9 controller is unchanged. Hugging Face generate() manages Qwen3.5's
hybrid attention/DeltaNet cache; a custom LogitsProcessor changes each step's
actual sampling distribution. The optional repetition-stop is separate.
"""
import os
import queue
import threading
import time
import traceback
from collections import deque

_model_lock = threading.Lock()
_generation_lock = threading.Lock()
_model = None
_tokenizer = None
_model_name = None


def load_model():
    global _model, _tokenizer, _model_name
    name = os.getenv('TP_MODEL_PATH', 'Qwen/Qwen3.5-2B')
    with _model_lock:
        if _model is not None and _model_name == name:
            return _model, _tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            name, device_map='auto', torch_dtype='auto', trust_remote_code=False
        ).eval()
        _model, _tokenizer, _model_name = model, tokenizer, name
        return model, tokenizer


def _entropy_of_top5(scores):
    import torch
    log_p = torch.log_softmax(scores.float(), dim=-1)
    top = torch.topk(log_p, k=min(5, log_p.shape[-1])).values
    p = top.exp()
    p = p / p.sum()
    h = -(p * torch.log(p + 1e-12)).sum().item()
    return h, (p[0] - p[1]).item() if p.numel() > 1 else p[0].item(), p[0].item()


class M9LogitsProcessor:
    """M9 benchz-aligned PD: same normalized top-5 H and 5% candidate cutoff."""
    def __init__(self, state):
        self.state = state
        self.prev_h = None
        self.t_base = float(os.getenv('TP_M9_T_BASE', '0.50'))
        self.h_target = float(os.getenv('TP_M9_H_TARGET', '1.20'))
        self.alpha = float(os.getenv('TP_M9_ALPHA', '0.25'))
        self.beta = float(os.getenv('TP_M9_BETA', '0.15'))
        self.t_min = float(os.getenv('TP_M9_T_MIN', '0.01'))
        self.t_max = float(os.getenv('TP_M9_T_MAX', '1.20'))
        if not 0 < self.t_min <= self.t_max:
            raise ValueError('Invalid M9 temperature limits')

    def __call__(self, input_ids, scores):
        import torch
        logits = scores[0].float()
        h, margin, top1 = _entropy_of_top5(logits)
        delta = h - self.prev_h if self.prev_h is not None else 0.0
        temp = max(self.t_min, min(self.t_max,
            self.t_base + self.alpha * (h - self.h_target) - self.beta * delta))
        self.prev_h = h
        scaled = logits / temp
        probabilities = torch.softmax(scaled, dim=-1)
        threshold = probabilities.max() * 0.05
        scaled = scaled.masked_fill(probabilities < threshold, float('-inf'))
        self.state['metric'] = {
            'h_t': round(h, 3), 't_t': round(temp, 3),
            'margin': round(margin, 3), 'top1': round(top1, 3),
            'temperature_applied': round(temp, 4),
            'temperature_next': None,
            'metric_source': 'm9_benchz_normalized_top5'
        }
        return scaled.unsqueeze(0).to(dtype=scores.dtype)



CONTROL_MODES = (
    'M1_Greedy', 'M2_Fixed_Low', 'M3_Fixed_Standard',
    'M4_Entropy_Threshold', 'M5_Mirostat_v2', 'M9_Adaptive_Hybrid'
)


class ComparisonLogitsProcessor:
    """benchz M1-M5 sampler settings with measured per-step UI metrics."""
    def __init__(self, state, control):
        self.state = state
        self.control = control
        self.prev_probs = None
        self.prev_input_len = 0
        self.mu = 10.0  # benchz Mirostat v2 initial mu = 2*tau

    def __call__(self, input_ids, scores):
        import torch
        logits = scores[0].float()
        h_top5, margin, top1 = _entropy_of_top5(logits)
        control = self.control
        if control == 'M1_Greedy':
            # do_sample=False handles argmax. No temperature applies to greedy.
            temperature = 0.0
            adjusted = logits
        elif control in ('M2_Fixed_Low', 'M3_Fixed_Standard'):
            temperature = 0.2 if control == 'M2_Fixed_Low' else 0.7
            adjusted = logits / temperature
        elif control == 'M4_Entropy_Threshold':
            temperature = 0.05 if h_top5 < 0.5 else (0.50 if h_top5 < 1.5 else 0.90)
            adjusted = logits / temperature
        elif control == 'M5_Mirostat_v2':
            # Mirror benchz: adapt mu to surprise of the *previous* sampled token.
            length = input_ids.shape[1]
            if self.prev_probs is not None and length > self.prev_input_len:
                last_id = int(input_ids[0, -1].item())
                last_prob = self.prev_probs[last_id].item()
                import math
                surprise = -math.log2(max(last_prob, 1e-12))
                self.mu -= 0.1 * (surprise - 5.0)
            temperature = 0.7
            adjusted = logits / temperature
            probs = torch.softmax(adjusted, dim=-1)
            # benchz M5 reports full-vocabulary entropy, unlike top-5 M1-M4/M9.
            full_h = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
            margin, top1 = _entropy_of_top5(adjusted)[1:]
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            surprises = -torch.log2(sorted_probs.clamp_min(1e-12))
            k = max(1, int((surprises <= self.mu).sum().item()))
            adjusted = adjusted.clone()
            adjusted[sorted_indices[k:]] = float('-inf')
            self.prev_probs = probs.detach()
            self.prev_input_len = length
            h_top5 = full_h
        else:
            raise ValueError(f'Unknown control mode: {control}')
        self.state['metric'] = {
            'h_t': round(h_top5, 3), 't_t': round(temperature, 3),
            'margin': round(margin, 3), 'top1': round(top1, 3),
            'temperature_applied': temperature, 'temperature_next': None,
            'metric_source': 'benchz_full_vocab_m5' if control == 'M5_Mirostat_v2' else 'benchz_top5_observed',
            'control': control
        }
        return adjusted.unsqueeze(0).to(dtype=scores.dtype)

class RepetitionStop:
    """Optional end-of-generation guard; does not modify logits or M9 gains."""
    def __init__(self, state, window=150):
        self.state = state
        self.window = window

    def __call__(self, input_ids, scores=None, **kwargs):
        return bool(self.state.get('stop', False))


def _has_repeated_tail(text):
    # Detect an identical 35-120-character passage repeated three times.
    text = text.rstrip()
    for n in range(35, min(120, len(text) // 3) + 1):
        if text[-n:] == text[-2*n:-n] == text[-3*n:-2*n]:
            return True
    return False


def _yield_generate(prompt, mode, model, tokenizer, control):
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer
    started = time.perf_counter()
    supports_thinking = 'qwen3' in os.getenv('TP_MODEL_PATH', 'Qwen/Qwen3.5-2B').lower()
    thinking = mode == 'reasoning' and supports_thinking
    if mode == 'reasoning' and not supports_thinking:
        yield 'stream_status', {'stage': 'reasoning_unsupported', 'message': '현재 모델은 Thinking을 지원하지 않습니다.', 'mode': mode}
    messages = [
        {'role': 'system', 'content': '답변은 질문의 언어로 작성하세요. 핵심 내용에 정확하고 간결하게 답하세요. 이미 말한 문장이나 의미를 반복하지 말고 충분히 답했다면 마무리하세요.'},
        {'role': 'user', 'content': prompt}
    ]
    if getattr(tokenizer, 'chat_template', None):
        kw = dict(tokenize=True, add_generation_prompt=True, return_tensors='pt')
        if supports_thinking:
            kw['enable_thinking'] = thinking
        input_ids = tokenizer.apply_chat_template(messages, **kw)
    else:
        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    # Transformers 5.x may return a BatchEncoding rather than a Tensor from
    # apply_chat_template(return_tensors='pt'). model.generate(input_ids=...)
    # requires the actual rank-2 token-ID tensor, not its wrapper.
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids.input_ids
    elif isinstance(input_ids, dict):
        input_ids = input_ids['input_ids']
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f'Unexpected chat-template output type: {type(input_ids).__name__}')
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f'Expected input_ids shape [1, sequence], got {tuple(input_ids.shape)}')
    input_ids = input_ids.to(device=model.device, dtype=torch.long)
    if control not in CONTROL_MODES:
        raise ValueError(f"Unsupported sampling controller: {control}")
    events = queue.Queue(maxsize=2048)
    state = {'metric': None, 'stop': False}
    done = object()

    class TokenStreamer(BaseStreamer):
        def __init__(self):
            self.first = True
            self.step = 0

        def put(self, value):
            ids = value.detach().flatten().tolist()
            if self.first:
                self.first = False
                # generate() streams the input prompt before any generated token.
                return
            for token_id in ids:
                self.step += 1
                events.put(('token', int(token_id), self.step, dict(state.get('metric') or {})))

        def end(self):
            events.put(('end', None, self.step, {}))

    max_tokens_default = '768' if thinking else '256'
    max_tokens = max(1, int(os.getenv('TP_MAX_NEW_TOKENS_REASONING' if thinking else 'TP_MAX_NEW_TOKENS', max_tokens_default)))
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    repetition_guard = os.getenv('TP_REPEAT_STOP', '1') == '1'
    controller = M9LogitsProcessor(state) if control == 'M9_Adaptive_Hybrid' else ComparisonLogitsProcessor(state, control)
    is_greedy = control == 'M1_Greedy'

    def worker():
        try:
            with torch.inference_mode():
                model.generate(
                    input_ids=input_ids, max_new_tokens=max_tokens,
                    do_sample=not is_greedy, temperature=1.0, top_p=1.0, top_k=0,
                    logits_processor=LogitsProcessorList([controller]),
                    stopping_criteria=StoppingCriteriaList([RepetitionStop(state)]),
                    streamer=TokenStreamer(), eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    use_cache=True
                )
        except Exception as exc:
            events.put(('error', traceback.format_exc(), 0, {}))
        finally:
            events.put((done, None, 0, {}))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    in_reasoning = thinking
    reasoning_events = 0
    output_text = ''
    while True:
        kind, token_id, step, metric = events.get()
        if kind is done:
            break
        if kind == 'error':
            raise RuntimeError(token_id)
        if kind != 'token':
            continue
        token = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if token in ('<|im_end|>', '<|endoftext|>', '<|im_start|>'):
            continue
        if token == '<think>' or token == '<|begin_of_thought|>':
            in_reasoning = True
            continue
        if token == '</think>' or token == '<|end_of_thought|>':
            in_reasoning = False
            continue
        if in_reasoning:
            reasoning_events += 1
            if reasoning_events == 1 or reasoning_events % 25 == 0:
                yield 'reasoning_progress', {
                    'stage': 'reasoning', 'events': reasoning_events,
                    'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                    'mode': mode
                }
            continue
        if not token or token.startswith('<|'):
            continue
        output_text += token
        payload = {
            'step': step, 'token': token, 'mode': mode, 'control': control,
            'upstream_elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
            'upstream_interval_ms': None
        }
        payload.update(metric)
        yield 'token', payload
        if repetition_guard and len(output_text) > 140 and _has_repeated_tail(output_text):
            state['stop'] = True
            yield 'stream_status', {'stage': 'repetition_stopped', 'message': '반복 문장 감지: 불필요한 토큰 생성을 종료했습니다.', 'mode': mode}
    thread.join(timeout=1)


def generate(prompt, mode='normal', control='M9_Adaptive_Hybrid'):
    if control not in CONTROL_MODES:
        raise ValueError(f'Unsupported sampling controller: {control}')
    model, tokenizer = load_model()
    with _generation_lock:
        yield from _yield_generate(prompt, mode, model, tokenizer, control)
