import os
from tqdm import tqdm
from pathlib import Path
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, CutSet
from lhotse import load_manifest_lazy
from lhotse import Fbank, FbankConfig, CutSet
import argparse
import pandas as pd


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

        # 2) Build RecordingSet
        recordings = []
        print("[INFO] Start to load in Recording Set...")
        for p in tqdm(wav_files):
            p = Path(p)
            assert p.exists(), f"Missing file: {p}"
            rec = Recording.from_file(p, recording_id=p.stem)  # id defaults to filename stem
            recordings.append(rec)
        recordings = RecordingSet.from_recordings(recordings)

        # 3) Build SupervisionSet (one segment per file, full duration)
        supervisions = []
        print("[INFO] Start to load in Supervision Set...")
        for rec, text in tqdm(zip(recordings, transcripts)):
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
        recordings_path   = out_dir + "/recordings.jsonl.gz"
        supervisions_path = out_dir + "/supervisions.jsonl.gz"
        recordings.to_file(recordings_path)
        supervisions.to_file(supervisions_path)

        # 5) Create CutSet and save (this is what you’ll load for raw)
        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        cuts_path = out_dir + "/cuts.jsonl.gz"
        cuts.to_file(cuts_path)

    def convert_to_fbank(self):
        cuts = load_manifest_lazy(self.output_dir + "/cuts.jsonl.gz")
        fbank = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80))
        cuts = cuts.compute_and_store_features(
            extractor=fbank,
            storage_path=self.output_dir + "/feats",       # directory for .llc feature files
            num_jobs=4                  # parallelism
        )
        cuts.to_file(self.output_dir + "/cuts_with_feats.jsonl.gz")

def get_parser():
    parser = argparse.ArgumentParser(description="Sherpa Data Preparation")
    parser.add_argument("--data_path", type=str, default="train.csv", help="Path to raw data")
    parser.add_argument("--output_dir", type=str, default="manifests", help="Directory to save manifests")
    parser.add_argument("--is_exist", type=int, default=0, help="Whether check audios are existing")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"raw data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")

    assert os.path.exists(args.data_path), f"raw data file {args.data_path} does not exist."
    raw_data = pd.read_csv(args.data_path, encoding='utf-8')
    print("[INFO] Load raw data.......")

    list_records = raw_data["path\ttranscript\tduration"].tolist()
    list_transcripts = []
    list_audios = []
    for element in list_records:
        parts = element.split('\t')
        if os.path.exists(parts[0]):
            list_audios.append(parts[0])
            list_transcripts.append(parts[1])
    print(f"[INFO] Number of audio files: {len(list_audios)}")
    if args.is_exist == 0:
        DataPreparation(list_audios, list_transcripts, args.output_dir)
    else:
        print("[INFO] Len of all list_records: ", len(list_records))
        print("[INFO] Len of existing audios: ", len(list_audios))

if __name__ == "__main__":
    main()
    # SAMPLE: python prepare_data.py --data_path='/data/audio_data/valid.csv' --output_dir='/data/audio_data/valid_feats'