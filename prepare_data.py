import k2
import lhotse
import os
from tqdm import tqdm
from pathlib import Path
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, CutSet
from lhotse import load_manifest_lazy
from lhotse import Fbank, FbankConfig, CutSet

class DataPreparation(object):
    def __init__(self, list_audios, list_transcripts, output_dir="manifests"):
        """
        Args:
            list_audios: a list of paths to audio files
            list_transcripts: a list of corresponding transcripts
            output_dir: directory to save the manifests
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print("[INFO] Start to load in Recording Set...")
        self.load_in_recording_set(list_audios, list_transcripts)
        print("[INFO] Load in Recording Set successfully!")
        self.convert_to_fbank()
        print("[INFO] Convert to FBank successfully!")

    def load_in_recording_set(self, wav_files, transcripts):
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2) Build RecordingSet
        recordings = []
        for p in wav_files:
            p = Path(p)
            assert p.exists(), f"Missing file: {p}"
            rec = Recording.from_file(p, recording_id=p.stem)  # id defaults to filename stem
            recordings.append(rec)
        recordings = RecordingSet.from_recordings(recordings)

        # 3) Build SupervisionSet (one segment per file, full duration)
        supervisions = []
        for rec, text in zip(recordings, transcripts):
            assert isinstance(text, str) and text.strip(), f"Empty transcript for {rec.id}"
            supervisions.append(
                SupervisionSegment(
                    id=f"{rec.id}-seg0",
                    recording_id=rec.id,
                    start=0.0,
                    duration=rec.duration,
                    text=text.strip(),
                    # Optional extras:
                    # speaker="spk1",
                    # language="vi",
                )
            )
        supervisions = SupervisionSet.from_segments(supervisions)

        # 4) Save manifests as .jsonl.gz
        recordings_path   = out_dir / "recordings.jsonl.gz"
        supervisions_path = out_dir / "supervisions.jsonl.gz"
        recordings.to_file(recordings_path)
        supervisions.to_file(supervisions_path)

        # 5) Create CutSet and save (this is what you’ll load for training)
        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        cuts_path = out_dir / "cuts_train.jsonl.gz"
        cuts.to_file(cuts_path)

    def convert_to_fbank(self):
        cuts = load_manifest_lazy(self.output_dir + "/cuts_train.jsonl.gz")
        fbank = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80))
        cuts = cuts.compute_and_store_features(
            extractor=fbank,
            storage_path="feats",       # directory for .llc feature files
            num_jobs=4                  # parallelism
        )
        cuts.to_file(self.output_dir + "/cuts_train_with_feats.jsonl.gz")