import torch
from generator import Generator
from utils.generation_utils import (
    get_input_encoding,
    get_output_texts,
    get_templated_prompt,
    unpad_output_texts,
)
from utils.trajectory import Trajectory
from utils.reward_utils import (
    compute_scores,
)
from engine.models.llm import LLM


class EarlyStopBon(Generator):
    def generate(self, prompt: str, prompt_dict: dict | None = None):
        self.prepare_generation(prompt_dict)
        self.clock.reset()
        self.clock.start()
        self.prompt = prompt
        self.templated_prompt = get_templated_prompt(
            prompt, self.args.llm_name, self.generation_tokenizer
        )

        sampling_num = 4
        sampling_size = int(self.args.batch_size / sampling_num)

        variances = []
        for __ in range(sampling_num):
            templated_prompts = [self.templated_prompt] * sampling_size
            batch_encoding = get_input_encoding(
                templated_prompts,
                self.generation_model,
                self.generation_tokenizer
            )

            self.clock.stop("tokenization")
            self.clock.start()

            input_length = batch_encoding.input_ids.shape[-1]
            max_all_tokens = min(self.args.max_tokens,
                                 self.args.max_gen_tokens + input_length)

            full_generation: torch.LongTensor = self.generation_model.generate(
                input_ids=batch_encoding.input_ids,
                attention_mask=batch_encoding.attention_mask,
                max_length=max_all_tokens,
                eos_token_id=self.terminators,
                pad_token_id=self.generation_tokenizer.pad_token_id,
                do_sample=True,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                temperature=self.args.temperature,
            )
            self.clock.stop("generation pass")

            self.clock.start()
            padded_output_texts = get_output_texts(
                full_generation,
                self.templated_prompt,
                self.generation_tokenizer,
                skip_special_tokens=False,
            )
            unpadded_output_texts = unpad_output_texts(
                padded_output_texts, self.stop_tokens
            )

            self.clock.stop("decoding")
            self.clock.start()
            reward_list = compute_scores(
                prompt,
                unpadded_output_texts,
                self.reward_model_name,
                self.reward_tokenizer,
                self.reward_model,
            )
            print(reward_list)

            self.clock.stop("reward pass")
            self.clock.start()
            for padded_output_text, unpadded_output_text, score in zip(
                padded_output_texts, unpadded_output_texts, reward_list
            ):
                trajectory = Trajectory(
                    self.prompt,
                    self.templated_prompt,
                    padded_output_text,
                    unpadded_output_text,
                    score,
                )
                self.trajectories.append(trajectory)

            variance = torch.std(torch.Tensor(reward_list))
            variances.append(variance)
            min_variance = min(variances)
            max_variance = max(variances)
            if (max_variance - min_variance) / min_variance > 0.1:
                break

        self.clock.stop("finish")
        self.post_generation()
