import k2
import re
import torch
import sentencepiece as spm
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import copy

from zipformer_model.train import get_parser, get_params, get_model, set_batch_count, get_adjusted_batch_count, update_averaged_model, save_checkpoint_with_global_batch_idx
from icefall_utils.utils import get_parameter_groups_with_lrs, MetricsTracker, make_pad_mask, count_trainable_parameters
from zipformer_model.optim import ScaledAdam, Eden
from icefall_utils.checkpoint import save_checkpoint
from zipformer_model.beam_search import (
    beam_search,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    fast_beam_search_one_best
)

from lhotse.features import Fbank, FbankConfig

import os
from tqdm import tqdm
import math
# Get environment variable safely (returns None if not set)

LOG_EPS = math.log(1e-10)
device = os.environ.get("DEVICE")

import jiwer
from tqdm import tqdm
import pandas as pd


transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ToLowerCase(),
    jiwer.RemoveKaldiNonWords(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


class Trainer(object):
    def __init__(self, folder_path, checkpoint_path, freeze_modules=[], is_streaming=False, decoding_method="greedy_search", max_duration=300, rank=0, use_ddp=False):
        parser = get_parser()
        args = parser.parse_args([])
        self.params = get_params()
        self.params.update(vars(args))
        self.params.max_duration = max_duration
        self.params.causal = is_streaming
        self.rank = rank
        self.use_ddp = use_ddp
        self.params.decoding_method = decoding_method
        self.params.max_sym_per_frame = 1
        self.token_table = k2.SymbolTable.from_file(folder_path + "/tokens.txt")
        self.params.blank_id = self.token_table["<blk>"]
        self.params.vocab_size = max(self.tokens()) + 1
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(folder_path + "/bpe.model")
        print("[INFO] Load model...")
        self.load_model(checkpoint_path)
        self.model.to(device)
        print("[INFO] Init optimizer, scaler, scheduler")
        self.freeze_modules = freeze_modules
        self.init_supporters()
        
        print("[INFO] Finish")
        
    def load_model(self, checkpoint_path):
        self.model = get_model(self.params)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # dst_state_dict = self.model.state_dict()
            # src_state_dict = checkpoint["model"]
            # for key in dst_state_dict.keys():
            #     src_key = key
            #     dst_state_dict[key] = src_state_dict.pop(src_key)
            # assert len(src_state_dict) == 0
            model_dict = self.model.state_dict()

            # Filter only keys that match in name AND shape
            compatible_state_dict = {
                k: v for k, v in checkpoint.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            self.model.load_state_dict(compatible_state_dict, strict=False)
            
        else:
            print("[INFO] Train from scratch")
        if self.rank == 0:
            self.model_avg = copy.deepcopy(self.model).to(torch.float64)

    def init_supporters(self):
        self.scaler = GradScaler(enabled=self.params.use_fp16, init_scale=1.0)
        self.optimizer = ScaledAdam(
                get_parameter_groups_with_lrs(self.model, lr=self.params.base_lr, include_names=True, freeze_modules=self.freeze_modules),
                lr=0.005,  # should have no effect
                clipping_scale=2.0,
            )
        self.scheduler = Eden(self.optimizer, self.params.lr_batches, self.params.lr_epochs)

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

    def normalize_batch(self, features):
        # Normalize per utterance
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True) + 1e-5
        features = (features - mean) / std
        return features


    def compute_loss(self, batch):
        feature = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)
        feature = self.normalize_batch(feature)
        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        batch_idx_train = self.params.batch_idx_train
        warm_step = self.params.warm_step

        texts = batch["supervisions"]["text"]
        y = self.texts_to_ids(texts)
        y = k2.RaggedTensor(y).to(device)

        with torch.set_grad_enabled(True):
            losses = self.model(
                x=feature,
                x_lens=feature_lens,
                y=y,
                prune_range=self.params.prune_range,
                am_scale=self.params.am_scale,
                lm_scale=self.params.lm_scale,
            )
            simple_loss, pruned_loss = losses[:2]

            s = self.params.simple_loss_scale
            # take down the scale on the simple loss from 1.0 at the start
            # to params.simple_loss scale by warm_step.
            simple_loss_scale = (
                s
                if batch_idx_train >= warm_step
                else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
            )
            pruned_loss_scale = (
                1.0
                if batch_idx_train >= warm_step
                else 0.1 + 0.9 * (batch_idx_train / warm_step)
            )

            loss = simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
        info = MetricsTracker()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info["frames"] = (feature_lens // self.params.subsampling_factor).sum().item()

        # Note: We use reduction=sum while computing the loss.
        info["loss"] = loss.detach().cpu().item()
        info["simple_loss"] = simple_loss.detach().cpu().item()
        info["pruned_loss"] = pruned_loss.detach().cpu().item()

        return loss, info
    
    def train_one_batch(self, batch_idx, batch):
        if batch_idx % 10 == 0:
            set_batch_count(self.model, get_adjusted_batch_count(self.params))

        self.params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        loss, loss_info = self.compute_loss(batch)
        # summary stats

        # NOTE: We use reduction==sum and loss is computed over utterances
        # in the batch and there is no normalization to it so far.
        self.scaler.scale(loss).backward()
        self.scheduler.step_batch(self.params.batch_idx_train)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if (self.params.batch_idx_train > 0 and self.params.batch_idx_train % self.params.average_period == 0):
            update_averaged_model(
                params=self.params,
                model_cur=self.model,
                model_avg=self.model_avg,
            )
    def save(self, checkpoint_folder, epoch):
        print("[INFO] Save checkpoint...")
        filename = f"{checkpoint_folder}/checkpoint-{epoch}.pt"
        print("[INFO] Checkpoint saved at: ", filename)
        save_checkpoint(
            filename=filename,
            model=self.model,
            model_avg=self.model_avg,
            params=self.params,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            sampler=None,
            rank=0,
        )
        return filename
    
    def train_one_epoch(self, train_dataloader, checkpoint_folder, epoch=0):
        tot_loss = MetricsTracker()
        cur_batch_idx = self.params.get("cur_batch_idx", 0)
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            if batch_idx % 10 == 0:
                set_batch_count(self.model, get_adjusted_batch_count(self.params))
            if batch_idx < cur_batch_idx:
                continue
            cur_batch_idx = batch_idx

            self.params.batch_idx_train += 1
            batch_size = len(batch["supervisions"]["text"])
            loss, loss_info = self.compute_loss(batch)
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / self.params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            self.scaler.scale(loss).backward()
            self.scheduler.step_batch(self.params.batch_idx_train)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if (self.params.batch_idx_train > 0 and self.params.batch_idx_train % self.params.average_period == 0):
                update_averaged_model(
                    params=self.params,
                    model_cur=self.model,
                    model_avg=self.model_avg,
                )
        print("[INFO] Save checkpoint...")
        filename = f"{checkpoint_folder}/checkpoint-{epoch}.pt"
        save_checkpoint(
            filename=filename,
            model=self.model,
            model_avg=self.model_avg,
            params=self.params,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            sampler=train_dataloader.sampler,
            rank=0,
        )

        return tot_loss

    def train(self, train_dataloader, num_epochs, checkpoint_folder):
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.tracks = []
        self.model.train()
        print("[INFO] Total trainable parameters: {}".format(
            count_trainable_parameters(self.model)))
        for epoch in range(num_epochs):
            epoch_loss = self.train_one_epoch(train_dataloader, checkpoint_folder, epoch)
            print(epoch_loss)
            self.tracks.append(epoch_loss)

    


    def test(self, batch):
        encoder_out, encoder_out_lens = self.encode_one_batch(batch)
        output_texts = self.decode(encoder_out, encoder_out_lens)
        output_texts = [o_.replace("▁", " ").lower() for o_ in output_texts]
        gt_texts = batch["supervisions"]["text"]
        wers = []
        for gt, hyp in zip(gt_texts, output_texts):
            wer_score = jiwer.wer(
                str(gt),
                str(hyp),
                truth_transform=transformation,
                hypothesis_transform=transformation
            )
            wers.append(wer_score)
        return wers 
    
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
        max_sym_per_frame = getattr(self.params, "max_sym_per_frame", 1)
        blank_penalty = getattr(self.params, "blank_penalty", 0)
        beam_size = getattr(self.params, "beam_size", 4)

        if self.params.decoding_method == "greedy_search" and max_sym_per_frame == 1:
            hyp_tokens = greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=blank_penalty,
            )
            for i in range(encoder_out.size(0)):
                hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
        elif self.params.decoding_method == "modified_beam_search":
            hyp_tokens = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=blank_penalty,
                beam=beam_size,
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
                    beam=beam_size,
                    max_contexts=self.params.max_contexts,
                    max_states=self.params.max_states,
                    blank_penalty=blank_penalty,
                )
                for i in range(encoder_out.size(0)):
                    hyps.append([self.token_table[idx] for idx in hyp_tokens[i]])
        elif self.params.decoding_method == "modified_beam_search":
            hyp_tokens = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=blank_penalty,
                beam=beam_size,
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
                        max_sym_per_frame=max_sym_per_frame,
                        blank_penalty=blank_penalty,
                    )
                elif self.params.decoding_method == "beam_search":
                    hyp = beam_search(
                        model=self.model,
                        encoder_out=encoder_out_i,
                        beam=beam_size,
                        blank_penalty=blank_penalty,
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