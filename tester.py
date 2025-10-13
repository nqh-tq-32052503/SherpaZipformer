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
MAX_DURATION = 300 # int(os.environ.get("MAX_DURATION"))
LOG_EPS = math.log(1e-10)
device = torch.device("cuda")


class Tester(object):
    def __init__(self, folder_path, checkpoint_path, is_streaming=False, decoding_method="greedy_search"):
        parser = get_parser()
        args = parser.parse_args([])
        self.params = get_params()
        self.params.update(vars(args))
        self.params.max_duration = MAX_DURATION
        self.params.causal = is_streaming
        self.params.decoding_method = decoding_method
        self.token_table = k2.SymbolTable.from_file(folder_path + "/tokens.txt")
        self.params.blank_id = self.token_table["<blk>"]
        self.params.vocab_size = max(self.tokens()) + 1
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
        elif "fast_beam_search" in self.params.decoding_method:
            decoding_graph = k2.trivial_graph(self.params.vocab_size - 1, device=device)
            if  self.params.decoding_method == "fast_beam_search_one_best":
                hyp_tokens = fast_beam_search_one_best(
                    model=self.model,
                    decoding_graph=decoding_graph,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    beam=self.params.beam,
                    max_contexts=self.params.max_contexts,
                    max_states=self.params.max_states,
                    blank_penalty=self.params.blank_penalty,
                )
                for i in range(encoder_out.size(0)):
                    hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
        elif    self.params.decoding_method == "modified_beam_search":
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

    def build_mel_transform(self, sr=16000, n_mels=80):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=400,            # 25 ms @ 16k
            win_length=400,
            hop_length=160,       # 10 ms @ 16k
            f_min=20.0,
            f_max=sr / 2,
            window_fn=torch.hann_window,
            n_mels=n_mels,
            center=True,
            power=2.0,            # magnitude^2, we'll log() later
            norm=None,
            mel_scale="htk",
        )

    @torch.no_grad()
    def wav_to_logmel(self, wav_16k: torch.Tensor, mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
        """
        wav_16k: 1D float32 tensor on the same device as mel, range [-1,1]
        returns: [T, 80] float32 log-mel
        """
        if wav_16k.ndim == 1:
            wav_16k = wav_16k.unsqueeze(0)  # [1, T]
        spec = mel(wav_16k) + LOG_EPS         # [1, n_mels, T']
        logmel = torch.log(spec).transpose(1, 2).squeeze(0).contiguous()  # [T', n_mels]
        return logmel

    def inference(self, audio_file):
        wav, sr = torchaudio.load(audio_file)  # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0).to(device=device, dtype=torch.float32)
        mel = self.build_mel_transform(sr=sr, n_mels=80).to(device)
        self.current_mel = mel
        self.current_wav = wav
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            feats = self.wav_to_logmel(wav, mel)  # [T, 80]
            feats_b = feats.unsqueeze(0)  # [1, T, 80]
            lengths = torch.tensor([feats_b.size(1)], device=device, dtype=torch.int32)
            batch = {"inputs" : feats_b, "supervisions" : {"num_frames" : lengths}}
            self.current_batch = batch
            encoder_out, encoder_out_lens = self.encode_one_batch(batch)
            self.current_encoder_out = {"features" : encoder_out, "lens" : encoder_out_lens}
            output_texts = self.decode(encoder_out, encoder_out_lens)
            return output_texts