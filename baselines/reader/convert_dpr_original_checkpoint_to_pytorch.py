# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
from pathlib import Path

import torch
from torch.serialization import default_restore_location

from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader


CheckpointState = collections.namedtuple(
    "CheckpointState", ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"]
)


import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union
import torch.nn.functional as F

from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader, DPRReaderOutput, DPRReaderTokenizer, BertModel

class UpdatedDPRReader(nn.Module):
    def __init__(self) -> None:
        super(UpdatedDPRReader, self).__init__()

        self.model = DPRReader(DPRConfig(**BertConfig.get_config_dict("bert-base-multilingual-uncased")[0]))
        bert_model = BertModel(BertConfig.from_pretrained("bert-base-multilingual-uncased"))

        self.model.span_predictor.encoder.bert_model = bert_model

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict=None,
    ) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        return outputs

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    print(f"Reading saved model from {model_file}")
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    return CheckpointState(**state_dict)


class DPRState:
    def __init__(self, src_file: Path):
        self.src_file = src_file

    def load_dpr_model(self):
        raise NotImplementedError

    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> "DPRState":
        if comp_type.startswith("c"):
            return DPRContextEncoderState(*args, **kwargs)
        if comp_type.startswith("q"):
            return DPRQuestionEncoderState(*args, **kwargs)
        if comp_type.startswith("r"):
            return DPRReaderState(*args, **kwargs)
        else:
            raise ValueError("Component type must be either 'ctx_encoder', 'question_encoder' or 'reader'.")


class DPRContextEncoderState(DPRState):
    def load_dpr_model(self):
        model = DPRContextEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-multilingual-uncased")[0]))
        print(f"Loading DPR biencoder from {self.src_file}")
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.ctx_encoder, "ctx_model."
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {"bert_model.embeddings.position_ids": model.ctx_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                state_dict[key] = value
        encoder.load_state_dict(state_dict)
        return model


class DPRQuestionEncoderState(DPRState):
    def load_dpr_model(self):
        model = DPRQuestionEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-multilingual-uncased")[0]))
        print(f"Loading DPR biencoder from {self.src_file}")
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.question_encoder, "question_model."
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {"bert_model.embeddings.position_ids": model.question_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                state_dict[key] = value
        encoder.load_state_dict(state_dict)
        return model


class DPRReaderState(DPRState):
    def load_dpr_model(self):
        updated_model = UpdatedDPRReader()
        print(f"Loading DPR reader from {self.src_file}")
        saved_state = load_states_from_checkpoint(self.src_file)
        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        state_dict = {
            "encoder.bert_model.embeddings.position_ids": updated_model.model.span_predictor.encoder.bert_model.embeddings.position_ids
        }

        for key, value in saved_state.model_dict.items():
            if key.startswith("encoder.") and not key.startswith("encoder.encode_proj"):
                key = "encoder.bert_model." + key[len("encoder.") :]
            state_dict[key] = value

        # print(state_dict.keys())
        # print(state_dict["encoder.bert_model.pooler.dense.weight"].shape)
        # print(state_dict["encoder.bert_model.encoder.layer.11.output.dense.weight"].shape)
        # print(state_dict["encoder.bert_model.pooler.dense.bias"].shape)
        # # print(model.span_predictor.state_dict().keys())
        # print(model)
        updated_model.model.span_predictor.load_state_dict(state_dict)
        return updated_model


def convert(comp_type: str, src_file: Path, dest_dir: Path):
    # dest_dir = Path(dest_dir)
    # dest_dir.mkdir(exist_ok=True)

    dpr_state = DPRState.from_type(comp_type, src_file=src_file)
    model = dpr_state.load_dpr_model()
    torch.save(model, dest_dir)# model.save_pretrained(pytorch_dump_path)

    # Verify that we can load the checkpoint.
    model = torch.load(dest_dir)  # sanity check
    print(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--type", type=str, help="Type of the component to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    )
    parser.add_argument(
        "--src",
        type=str,
        help=(
            "Path to the dpr checkpoint file. They can be downloaded from the official DPR repo"
            " https://github.com/facebookresearch/DPR. Note that in the official repo, both encoders are stored in the"
            " 'retriever' checkpoints."
        ),
    )
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model directory.")
    args = parser.parse_args()

    src_file = Path(args.src)
    dest_dir = f"converted-{src_file.name}" if args.dest is None else args.dest
    dest_dir = Path(dest_dir)
    assert src_file.exists()
    assert (
        args.type is not None
    ), "Please specify the component type of the DPR model to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    convert(args.type, src_file, dest_dir)