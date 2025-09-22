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
)
from zipformer_model.decode import get_parser 

import os
from tqdm import tqdm
# Get environment variable safely (returns None if not set)
MAX_DURATION = os.environ.get("MAX_DURATION")
device = torch.device("cuda")


class Tester(object):
    def __init__(self, folder_path, checkpoint_path):
        parser = get_parser()
        args = parser.parse_args([])
        self.params = get_params()
        self.params.update(vars(args))
        self.params.max_duration = MAX_DURATION
        self.token_table = k2.SymbolTable.from_file(folder_path + "/pseudo_data/tokens.txt")
        self.params.blank_id = self.token_table["<blk>"]
        self.params.vocab_size = max(self.tokens()) + 1
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(folder_path + "/pseudo_data/bpe.model")
        print("[INFO] Load model...")
        self.load_model(checkpoint_path)
        self.model.to(device)
        self.model.eval()

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
            # at entry, feature is (N, T, C)
    
            supervisions = batch["supervisions"]
            feature_lens = supervisions["num_frames"].to(device)
            x, x_lens = self.model.encoder_embed(feature, feature_lens)
            src_key_padding_mask = make_pad_mask(x_lens)
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
    
            encoder_out, encoder_out_lens = self.model.encoder(x, x_lens, src_key_padding_mask)
            encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
            return encoder_out, encoder_out_lens

    def decode(self, encoder_out, encoder_out_lens):
        print("[INFO] Decoding method=", self.params.decoding_method)
        hyps = []

        if self.params.decoding_method == "greedy_search" and self.params.max_sym_per_frame == 1:
            hyp_tokens = greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=self.params.blank_penalty,
            )
            for i in range(encoder_out.size(0)):
                hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
        elif self.params.decoding_method == "modified_beam_search":
            hyp_tokens = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=self.params.blank_penalty,
                beam=self.params.beam_size,
            )
            for i in range(encoder_out.size(0)):
                hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
        else:
            batch_size = encoder_out.size(0)

            for i in range(batch_size):
                # fmt: off
                encoder_out_i = encoder_out[i:i + 1, :encoder_out_lens[i]]
                # fmt: on
                if self.params.decoding_method == "greedy_search":
                    hyp = greedy_search(
                        model=self.model,
                        encoder_out=encoder_out_i,
                        max_sym_per_frame=self.params.max_sym_per_frame,
                        blank_penalty=self.params.blank_penalty,
                    )
                elif self.params.decoding_method == "beam_search":
                    hyp = beam_search(
                        model=self.model,
                        encoder_out=encoder_out_i,
                        beam=self.params.beam_size,
                        blank_penalty=self.params.blank_penalty,
                    )
                else:
                    raise ValueError(
                        f"Unsupported decoding method: {self.decoding_method}"
                    )
                hyps.append([self.token_table[idx] for idx in hyp])
        output_texts = []
        for hyp in hyps:
            output = "".join(h_ for h_ in hyp)
            output_texts.append(output)
        return output_texts
    def __call__(self, batch):
        encoder_out, encoder_out_lens = self.encode_one_batch(batch)
        output_texts = self.decode(encoder_out, encoder_out_lens)
        return output_texts