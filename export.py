# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
from pprint import pprint

import paddle
from paddlenlp.utils.log import logger

from paddlenlp.ops import FasterUNIMOText
from paddlenlp.transformers import UNIMOLMHeadModel, UNIMOTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="./model_final",
        type=str,
        help="The model name to specify the Pegasus to use. ",
    )
    parser.add_argument(
        "--export_output_dir", default="./inference_model", type=str, help="Path to save inference model of Pegasus. "
    )
    parser.add_argument("--topk", default=4, type=int, help="The number of candidate to procedure top_k sampling. ")
    parser.add_argument(
        "--topp", default=1.0, type=float, help="The probability threshold to procedure top_p sampling. "
    )
    parser.add_argument("--max_out_len", default=64, type=int, help="Maximum output length. ")
    parser.add_argument("--min_out_len", default=1, type=int, help="Minimum output length. ")
    parser.add_argument("--num_return_sequence", default=1, type=int, help="The number of returned sequence. ")
    parser.add_argument("--temperature", default=1.0, type=float, help="The temperature to set. ")
    parser.add_argument("--num_return_sequences", default=1, type=int, help="The number of returned sequences. ")
    parser.add_argument("--use_fp16_decoding", action="store_true", help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--decoding_strategy",
        default="beam_search",
        choices=["beam_search"],
        type=str,
        help="The main strategy to decode. ",
    )
    parser.add_argument("--num_beams", default=4, type=int, help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--diversity_rate", default=0.0, type=float, help="The diversity rate to procedure beam search. "
    )
    parser.add_argument(
        "--length_penalty",
        default=0.0,
        type=float,
        help="The exponential penalty to the sequence length in the beam_search strategy. ",
    )

    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)
    paddle.set_default_dtype("float16")

    model_name_or_path = args.model_name_or_path

    model = UNIMOLMHeadModel.from_pretrained(model_name_or_path)
    # import pdb; pdb.set_trace()
    model = model.to(dtype="float16")
    tokenizer = UNIMOTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    unimo = FasterUNIMOText(model=model, use_fp16_decoding=True, trans_out=True
                            # , decoding_lib='./libdecoding_op.so'
                            )

    # Set evaluate mode
    unimo.eval()

    # Convert dygraph model to static graph model
    unimo = paddle.jit.to_static(
        unimo,
        input_spec=[
            # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            # token_type_ids,
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            # attention_mask,
            paddle.static.InputSpec(shape=[None, None, None, None], dtype="float32"),
            # seq_len
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            # max_length=128,
            64,
            # min_length=0,
            10,
            # top_k=4,
            4,
            # top_p=0.0,
            1.0,
            # num_beams=4,
            1,
            # decode_strategy="sampling",
            "beam_search",
            # bos_token_id=None,
            tokenizer.cls_token_id,
            # eos_token_id=None,
            tokenizer.mask_token_id,
            # pad_token_id=
            None,
            # diversity_rate=
            0.0,
            # temperature=
            1.0,
            # num_return_sequences=
            1,
            # length_penalty=0.6,
            1.2,
            # early_stopping=False,
            # forced_eos_token_id=None,
            # position_ids=None,
        ],
    )

    # Save converted static graph model
    paddle.jit.save(unimo, os.path.join(args.export_output_dir, "unimo"))
    logger.info("unimo has been saved to {}.".format(args.export_output_dir))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    do_predict(args)
