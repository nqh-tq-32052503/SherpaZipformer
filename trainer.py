import k2
import re
import torch
import sentencepiece as spm
from torch.cuda.amp import GradScaler
import warnings
import copy

from zipformer_model.train import get_parser, get_params, get_model, set_batch_count, get_adjusted_batch_count, update_averaged_model, save_checkpoint_with_global_batch_idx
from icefall_utils.utils import get_parameter_groups_with_lrs, MetricsTracker, make_pad_mask
from zipformer_model.optim import ScaledAdam, Eden
from icefall_utils.checkpoint import save_checkpoint


import os
from tqdm import tqdm
# Get environment variable safely (returns None if not set)
MAX_DURATION = int(os.environ.get("MAX_DURATION"))
device = torch.device("cuda")

class Trainer(object):
    def __init__(self, folder_path, checkpoint_path, freeze_modules=[]):
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
        print("[INFO] Init optimizer, scaler, scheduler")
        self.freeze_modules = freeze_modules
        self.init_supporters()
        print("[INFO] Finish")
        
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

    def compute_loss(self, batch):
        feature = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

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

    def train_one_epoch(self, train_dataloader, checkpoint_folder, epoch=0):
        self.model.train()
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
        self.tracks = []
        for epoch in range(num_epochs):
            epoch_loss = self.train_one_epoch(train_dataloader, checkpoint_folder, epoch)
            print(epoch_loss)
            self.tracks.append(epoch_loss)

