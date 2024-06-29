import torch
import requests
import numpy as np
from zerollama.tasks.reranker.interface import RerankerInterface
from zerollama.tasks.reranker.protocol import RerankerResponse
from zerollama.tasks.reranker.collection import get_model_config_by_name, get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name


class Zhinao360Reranker(RerankerInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.model_name_or_path = get_pretrained_model_name(model_name=model_name,
                                                            local_files_only=local_files_only,
                                                            get_model_by_name=get_model_by_name)

        self.model = None
        self.tokenizer = None
        self.IGNORE_TOKEN_ID = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup

        config_setup()

        from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                model_max_length=1024,
                padding_side="right",
                use_fast=False,
                trust_remote_code=True,
                local_files_only=self.local_files_only
            )
            self.tokenizer.pad_token_id = self.tokenizer.eod_id
            config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                bf16=True,
                local_files_only=self.local_files_only
            )
            config.use_cache = False
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=config,
                trust_remote_code=True,
                local_files_only=self.local_files_only
            )
            self.model.linear.bfloat16()

            from transformers.trainer_pt_utils import LabelSmoother
            self.IGNORE_TOKEN_ID = LabelSmoother.ignore_index

        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            import traceback
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

        self.model.half()
        self.model = self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def compute_score(self, sentence_pairs, options=None):
        options = options or {}
        batch_size = options.get("batch_size", 32)

        all_scores = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            inputs = self.preprocess(sources=sentence_pairs, tokenizer=self.tokenizer, max_len=1024, device=self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        return RerankerResponse(model=self.model_name, vecs={"scores": np.array(all_scores)})

    @property
    def info(self):
        return self.model_info

    def preprocess(
            self,
            sources,
            tokenizer,
            max_len: int = 1024,
            system_message: str = "",
            device=None,
    ):
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
        answer_len = 64

        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
        nl_tokens = tokenizer('\n').input_ids
        _system = tokenizer('system').input_ids + nl_tokens
        _user = tokenizer('user').input_ids + nl_tokens
        _assistant = tokenizer('assistant').input_ids + nl_tokens

        # Apply prompt templates
        input_ids, targets = [], []
        for i, source in enumerate(sources):
            ## system_message
            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message, max_length=max_len - answer_len,
                                                      truncation=True).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [self.IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
            assert len(input_id) == len(target)

            ## query ans
            source = "\n\n".join(source)
            role = "<|im_start|>user"
            _input_id = tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids + nl_tokens + \
                        tokenizer(source, max_length=max_len - answer_len, truncation=True).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [self.IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [self.IGNORE_TOKEN_ID] * len(
                    tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids) + \
                          _input_id[
                          len(tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids) + 1:-2] + [
                              im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target

            ## label use placeholder 0; It will be masked later in the modeling_zhinao.py
            role = "<|im_start|>assistant"
            _input_id = tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids + nl_tokens + \
                        tokenizer("0", max_length=max_len - answer_len, truncation=True).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [self.IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [self.IGNORE_TOKEN_ID] * len(
                    tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids) + \
                          _input_id[
                          len(tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids) + 1:-2] + [
                              im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target

            assert len(input_id) == len(target)
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [self.IGNORE_TOKEN_ID] * (max_len - len(target))
            if len(input_id) > max_len:
                print("max_len_error")
                print(tokenizer.decode(input_id))

            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.int)
        # print(f"input_ids {input_ids.shape}")
        # print(f"targets {targets.shape}")

        return dict(
            input_ids=input_ids.to(device),
            labels=targets.to(device),
            attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device),
        )