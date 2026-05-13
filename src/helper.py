import re
import torch
import logging
from keys import mykey

# A dictionary to cache models and tokenizers to avoid reloading

global models
models = {}

def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger: 
        if mode == "error": logger.error(message)
        if mode == "warning": logger.warning(message)
        else: logger.info(message)
    if print_to_std: print(message + "\n")

class ModelCache:
    def __init__(self, model_name, use_vllm=False, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        if self.use_api == "openai":
            from openai import OpenAI
            self.api_account = self.args.get("api_account", "openai")
            self.client = OpenAI(api_key=mykey[self.api_account]) # Setup API key appropriately in keys.py
        elif self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", False)
                tensor_parallel_size = self.args.get("tensor_parallel_size", 1)
                max_num_seqs = self.args.get("batch_size", 256)
                self.model = LLM(model=self.model_name, enable_prefix_caching=enable_prefix_caching, tensor_parallel_size=tensor_parallel_size, max_num_seqs=max_num_seqs)
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.terminators = self._get_terminators()
            except Exception as e:
                raise RuntimeError(f"[{self.model_name}] vLLM failed to load: {e}")
        if not self.use_vllm and self.use_api != "openai":
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
            try:
                self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.model.eval()
            # Processor wraps a tokenizer; fall back to the inner tokenizer for token attrs
            _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            _tok.pad_token = _tok.eos_token
            _tok.pad_token_id = _tok.eos_token_id
            self.terminators = self._get_terminators()
    
    def _get_terminators(self):
        _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        terminators = [_tok.eos_token_id]
        for token in ["<|eot_id|>", "<end_of_turn>", "<|im_end|>", "</s>"]:
            tid = _tok.convert_tokens_to_ids(token)
            if tid and tid != _tok.unk_token_id:
                terminators.append(tid)
        return list(set(terminators))

    def generate(self, messages):

        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_tokens", 256)
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)

        if self.use_api == "openai":
            result = self.openai_generate(messages)
        elif self.use_vllm:
            result = self.vllm_generate(messages)
        else:
            result = self.huggingface_generate(messages)
        return result
    
    def _to_content_list(self, messages):
        """Convert plain string content to list-of-dicts format expected by processors like MedGemma."""
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            out.append({"role": m["role"], "content": content})
        return out

    def huggingface_generate(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            self._to_content_list(messages), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)

        raw_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        log_info(f"[{self.model_name}][RAW OUTPUT]:\n{raw_text}")
        print(f"[LLM RAW OUTPUT]:\n{raw_text}\n")
        # Strip Gemma 3 thinking block (<unused94>...<unused95>) if present
        # Complete block: remove everything between the tags
        stripped = re.sub(r'<[^>]+>.*?<[^>]+>', '', raw_text, flags=re.DOTALL).strip()
        # Partial block (thinking cut off, no closing tag): take everything after the last '>'
        if not stripped:
            after_tag = re.split(r'>[^>]*$', raw_text)
            stripped = after_tag[-1].strip() if len(after_tag) > 1 else raw_text.strip()
        response_text = stripped
        usage = {"input_tokens": input_len, "output_tokens": outputs.shape[-1] - input_len}
        log_info(f"[{self.model_name}][PARSED OUTPUT]: {response_text}")
        return response_text, None, usage
        
    def vllm_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, logprobs=self.top_logprobs, 
                                        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        outputs = self.model.generate(inputs, sampling_params)
        response_text = outputs[0].outputs[0].text
        logprobs = outputs[0].outputs[0].cumulative_logprob
        # TODO: If top_logprobs > 0, return logprobs of generation
        # if self.top_logprobs > 0: logprobs = outputs[0].outputs[0].logprobs
        usage = {"input_tokens": len(outputs[0].prompt_token_ids), "output_tokens": len(outputs[0].outputs[0].token_ids)}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        if self.top_logprobs == 0:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
        else:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        logprobs=True, 
                        top_logprobs=self.top_logprobs
                    )
        
        num_input_tokens = response["usage"]["prompt_tokens"]
        num_output_tokens = response["usage"]["completion_tokens"]
        response_text = response.choices[0].text.strip()
        log_probs = response.choices[0].logprobs.top_logprobs if self.top_logprobs > 0 else None
        
        log_info(f"[{self.model_name}][OUTPUT]: {response}")
        return response_text, log_probs, {"input_tokens": num_input_tokens, "output_tokens": num_output_tokens}


def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    if 'gpt' in model_name or 'o1' in model_name: use_api = "openai"
    
    model_cache = models.get(model_name, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **kwargs)
        models[model_name] = model_cache
    
    return model_cache.generate(messages)
