import k2
import re
import torch
import sentencepiece as spm
from zipformer_model.train import get_parser, get_params, get_model
from icefall_utils.utils import  make_pad_mask
from zipformer_model.beam_search import (
    beam_search,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    fast_beam_search_one_best
)
from zipformer_model.decode import get_parser 

import os
import math
from tqdm import tqdm
import torchaudio
# Get environment variable safely (returns None if not set)

LOG_EPS = math.log(1e-10)
device = torch.device(os.environ.get("DEVICE", "cuda"))


class Tester(object):
    def __init__(self, folder_path, checkpoint_path, is_streaming=False, decoding_method="greedy_search", max_duration=300, return_timestamps=True):
        parser = get_parser()
        args = parser.parse_args([])
        self.params = get_params()
        self.params.update(vars(args))
        self.params.max_duration = max_duration
        self.params.causal = is_streaming
        self.params.decoding_method = decoding_method
        self.token_table = k2.SymbolTable.from_file(folder_path + "/tokens.txt")
        self.params.blank_id = self.token_table["<blk>"]
        self.params.vocab_size = max(self.tokens()) + 1
        self.return_timestamps = return_timestamps
        print("[INFO] Return timestamps: ", self.return_timestamps)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(folder_path + "/bpe.model")
        print("[INFO] Load model...")
        self.load_model(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        print("[INFO] Decoding method=", self.params.decoding_method)
        torch.cuda.init()

    def load_model(self, checkpoint_path):
        self.model = get_model(self.params)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        dst_state_dict = self.model.state_dict()
        src_state_dict = checkpoint["model"]
        for key in dst_state_dict.keys():
            src_key = key
            dst_state_dict[key] = src_state_dict.pop(src_key)
        assert len(src_state_dict) == 0
        self.model.load_state_dict(dst_state_dict, strict=True)
        for param in self.model.parameters():
            param.requires_grad = False

    def tokens(self):
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        Caution:
          0 is not a token ID so it is excluded from the return value.
        """
        disambig_pattern = re.compile(r"^#\d+$")
        symbols = self.token_table.symbols
        ans = []
        for s in symbols:
            if not disambig_pattern.match(s):
                ans.append(self.token_table[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans

    def texts_to_ids(self, texts):
        ids = []
        for text in texts:
            upper_text = text.upper()
            ids.append(self.sp.encode(upper_text, out_type=int))
        return ids

    def encode_one_batch(self, batch):
        with torch.no_grad():
            feature = batch["inputs"]
            assert feature.ndim == 3
    
            feature = feature.to(device)
            feature.requires_grad = False
            # at entry, feature is (N, T, C)
    
            supervisions = batch["supervisions"]
            feature_lens = supervisions["num_frames"].to(device)
            if self.params.causal:
                # this seems to cause insertions at the end of the utterance if used with zipformer.
                pad_len = 30
                feature_lens += pad_len
                feature = torch.nn.functional.pad(
                    feature,
                    pad=(0, 0, 0, pad_len),
                    value=LOG_EPS,
                )
            x, x_lens = self.model.encoder_embed(feature, feature_lens)
            src_key_padding_mask = make_pad_mask(x_lens)
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
    
            encoder_out, encoder_out_lens = self.model.encoder(x, x_lens, src_key_padding_mask)
            encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
            return encoder_out, encoder_out_lens

    def decode(self, encoder_out, encoder_out_lens):
        hyps = []
        timestamps = []
        if self.params.decoding_method == "greedy_search" and self.params.max_sym_per_frame == 1:
            answers = greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=self.params.blank_penalty,
                return_timestamps=self.return_timestamps
            )
            if self.return_timestamps:
                hyp_tokens = answers.hyps
                timestamps.extend(answers.timestamps)
            else:
                hyp_tokens = answers
            for i in range(encoder_out.size(0)):
                hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
            
        elif self.params.decoding_method == "modified_beam_search":
            answers = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=self.params.blank_penalty,
                beam=self.params.beam_size,
                return_timestamps=self.return_timestamps
            )
            if self.return_timestamps:
                hyp_tokens = answers.hyps
                timestamps.extend(answers.timestamps)
            else:
                hyp_tokens = answers
            for i in range(encoder_out.size(0)):
                hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
        elif "fast_beam_search" in self.params.decoding_method:
            decoding_graph = k2.trivial_graph(self.params.vocab_size - 1, device=device)
            if  self.params.decoding_method == "fast_beam_search_one_best":
                answers = fast_beam_search_one_best(
                    model=self.model,
                    decoding_graph=decoding_graph,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    beam=self.params.beam,
                    max_contexts=self.params.max_contexts,
                    max_states=self.params.max_states,
                    blank_penalty=self.params.blank_penalty,
                    return_timestamps=self.return_timestamps
                )
                if self.return_timestamps:
                    hyp_tokens = answers.hyps
                    timestamps.extend(answers.timestamps)
                else:
                    hyp_tokens = answers
                for i in range(encoder_out.size(0)):
                    hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
        elif self.params.decoding_method == "modified_beam_search":
            answers = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=self.params.blank_penalty,
                beam=self.params.beam_size,
                return_timestamps=self.return_timestamps
            )
            if self.return_timestamps:
                hyp_tokens = answers.hyps
                timestamps.extend(answers.timestamps)
            else:
                hyp_tokens = hyps
            for i in range(encoder_out.size(0)):
                hyps.append([self.token_table[idx] for idx in hyp_tokens[i]]) 
        else:
            batch_size = encoder_out.size(0)

            for i in range(batch_size):
                # fmt: off
                encoder_out_i = encoder_out[i:i + 1, :encoder_out_lens[i]]
                # fmt: on
                if self.params.decoding_method == "greedy_search":
                    answer = greedy_search(
                        model=self.model,
                        encoder_out=encoder_out_i,
                        max_sym_per_frame=self.params.max_sym_per_frame,
                        blank_penalty=self.params.blank_penalty,
                        return_timestamps=self.return_timestamps
                    )
                    if self.return_timestamps:
                        hyp = answer.hyps[0]
                        timestamps.extend(answer.timestamps[0])
                    else:
                        hyp = answer
                elif self.params.decoding_method == "beam_search":
                    answer = beam_search(
                        model=self.model,
                        encoder_out=encoder_out_i,
                        beam=self.params.beam_size,
                        blank_penalty=self.params.blank_penalty,
                        return_timestamps=self.return_timestamps
                    )
                    if self.return_timestamps:
                        hyp = answer.hyps[0]
                        timestamps.extend(answer.timestamps[0])
                    else:
                        hyp = answer
                else:
                    raise ValueError(
                        f"Unsupported decoding method: {self.decoding_method}"
                    )
                hyps.append([self.token_table[idx] for idx in hyp])
        final_results = []
        for index, hyp in enumerate(hyps):
            element = {"transcript" : "",
                       "items" : []}
            transcript = "".join(h_ for h_ in hyp)
            transcript = str(transcript).replace("▁", " ").lower()
            element["transcript"] = transcript
            if self.return_timestamps:
                timestamp = timestamps[index]
                token_id = 0
                for h_, t_ in zip(hyp, timestamp):
                    element["items"].append({"id" : str(token_id),
                                             "segment" : str(h_).replace("▁", " ").lower(),
                                             "start_time" : str(t_),
                                             "end_time" : ""})
                    token_id += 1
            final_results.append(element)
        return final_results
    def __call__(self, batch):
        encoder_out, encoder_out_lens = self.encode_one_batch(batch)
        final_results = self.decode(encoder_out, encoder_out_lens)
        return final_results
