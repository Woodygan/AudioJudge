from huggingface_hub import hf_hub_download
import torch
import os
import soundfile as sf
import random
import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.signal import resample_poly

from datasets import load_dataset
from dataclasses import dataclass
import random
import time
import sentencepiece
import torch
import sphn
from moshi.client_utils import log
from moshi.conditioners import ConditionAttributes, ConditionTensors
from moshi.models import loaders, MimiModel, LMModel, LMGen


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_condition_tensors(
    model_type: str, lm: LMModel, batch_size: int, cfg_coef: float
) -> ConditionTensors:
    condition_tensors = {}
    if lm.condition_provider is not None:
        conditions: list[ConditionAttributes] | None = None
        if model_type == "hibiki":
            conditions = [
                ConditionAttributes(text={"description": "very_good"}, wav={})
            ] * batch_size
            if cfg_coef != 1.0:
                # Extending the conditions with the negatives for the CFG.
                conditions += [
                    ConditionAttributes(text={"description": "very_bad"}, wav={})
                ] * batch_size
        else:
            raise RuntimeError(
                f"Model expects conditioning but model type {model_type} is not supported."
            )
        assert conditions is not None
        prepared = lm.condition_provider.prepare(conditions)
        condition_tensors = lm.condition_provider(prepared)
    return condition_tensors


@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

    def __init__(
        self,
        model_type: str,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        cfg_coef: float,
        device: str | torch.device,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size, cfg_coef)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)

    def run(self, in_pcms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_pcms = []
        out_text_tokens = []
        log("info", "starting the inference loop")
        start_time = time.time()
        ntokens = 0
        for i, chunk in enumerate(in_pcms.split(1920, dim=2)):
            if chunk.shape[-1] != 1920:
                break
            codes = self.mimi.encode(chunk)
            if i == 0:
                # Ensure that the first slice of codes is properly seen by the transformer
                # as otherwise the first slice is replaced by the initial tokens.
                tokens = self.lm_gen.step(codes)
                assert tokens is None
            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
            out_pcm = self.mimi.decode(tokens[:, 1:])
            out_text_tokens.append(tokens[:, 0])
            out_pcms.append(out_pcm)
            ntokens += 1
        dt = time.time() - start_time
        log(
            "info",
            f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step",
        )
        out_pcms = torch.cat(out_pcms, dim=2)
        out_text_tokens = torch.cat(out_text_tokens, dim=1)
        return out_pcms, out_text_tokens


class InferenceHandler:
    def __init__(self):
        device = "cuda"

        self.device = device
        self.batch_size = 1
        log("info", "moshi loaded")

    def _load_model(self):
        self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            loaders.DEFAULT_REPO,
            moshi_weights=None,
            mimi_weights=None,
            tokenizer=None,
            config_path=None,
        )
        log("info", "loading mimi")
        self.mimi = self.checkpoint_info.get_mimi(device=self.device)
        log("info", "mimi loaded")
        self.text_tokenizer = self.checkpoint_info.get_text_tokenizer()
        log("info", "loading moshi")
        self.lm = self.checkpoint_info.get_moshi(device=self.device)

    @torch.no_grad()
    def inference(self, inpath: str, outpath: str):
        self._load_model()
        log("info", f"loading input file {inpath}")
        in_pcms, _ = sphn.read(inpath, sample_rate=self.mimi.sample_rate)
        in_pcms = torch.from_numpy(in_pcms).to(device=self.device)
        in_pcms = in_pcms[None, 0:1].expand(self.batch_size, -1, -1)

        state = InferenceState(
            self.checkpoint_info.model_type,
            self.mimi,
            self.text_tokenizer,
            self.lm,
            self.batch_size,
            cfg_coef=1.0,
            device=self.device,
        )
        out_pcms, out_text_tokens = state.run(in_pcms)
        log("info", f"out-pcm: {out_pcms.shape}, out-text: {out_text_tokens.shape}")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if self.batch_size == 1:
            sphn.write_wav(
                outpath, out_pcms[0, 0].cpu().numpy(), sample_rate=self.mimi.sample_rate
            )
        else:
            raise NotImplementedError()


@torch.no_grad()
def experiment(output_dir):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("-----------------------------")

    # inference handler from Ink
    handler = InferenceHandler()

    # load dataset
    with open("data/questions1_shuffled_id.json", "r") as f:
        tmp_dataset = json.load(f)
    print("len(dataset):", len(tmp_dataset))

    ids = [i for i in range(len(tmp_dataset))]
    random.shuffle(ids)

    for i in tqdm(ids):
        output_file = f"{output_dir}/audio/{i}.wav"
        # check if the file already exists
        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists.")
            continue

        # Load the input WAV file
        input_file = f"data/questions1_kokoro_wav/{i}.kokoro.wav"

        # run inference and save the output wav file
        handler.inference(input_file, outpath=output_file)
        print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    args = parser.parse_args()
    experiment(args.output_dir)

    # usage: python inference_moshi.py --output_dir experiments/advvoiceq1/moshi


if __name__ == "__main__":
    main()
