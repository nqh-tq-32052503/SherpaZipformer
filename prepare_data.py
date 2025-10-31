import os
from tqdm import tqdm
from pathlib import Path
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, CutSet
from lhotse import load_manifest_lazy, load_manifest, fix_manifests, validate_recordings_and_supervisions
from lhotse import Fbank, FbankConfig, CutSet
import argparse
import pandas as pd


class DataPreparation(object):
    def __init__(self, list_audios, list_transcripts, output_dir="manifests", extract_fbank=True):
        """
        Args:
            list_audios: list đường dẫn WAV
            list_transcripts: list transcript tương ứng (cùng chiều)
            output_dir: thư mục lưu manifests/feats
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print("[INFO] Build Recording/Supervision...")
        self.build_and_save_manifests(list_audios, list_transcripts)
        print("[INFO] Build manifests done.")

        print("[INFO] Compute + store FBanks...")
        if extract_fbank:
            self.compute_and_store_fbank()
        print("[INFO] FBanks done.")

    def build_and_save_manifests(self, wav_files, transcripts):
        out_dir = self.output_dir

        # 1) Build RecordingSet (eager)
        recs = []
        for p in tqdm(wav_files, desc="RecordingSet"):
            p = Path(p)
            assert p.exists(), f"Missing file: {p}"
            # id = stem để khớp với sups
            recs.append(Recording.from_file(p, recording_id=p.stem))
        recordings = RecordingSet.from_recordings(recs)

        # 2) Build SupervisionSet (mỗi file 1 segment full duration)
        assert len(wav_files) == len(transcripts), "Audio/transcript length mismatch"
        sups = []
        for rec, text in tqdm(zip(recordings, transcripts), total=len(transcripts), desc="SupervisionSet"):
            text = str(text).strip()
            assert text, f"Empty transcript for {rec.id}"
            sups.append(
                SupervisionSegment(
                    id=f"{rec.id}-seg0",
                    recording_id=rec.id,
                    start=0.0,
                    duration=rec.duration,
                    text=text,
                )
            )
        supervisions = SupervisionSet.from_segments(sups)

        # 3) Sửa và validate (rất nên làm)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        # 4) Tạo CutSet (eager) và TRIM THEO SUPS (eager)
        cuts_raw = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        cuts_trim = cuts_raw.trim_to_supervisions(keep_overlapping=False).to_eager()

        # 5) Lưu tất cả dưới dạng EAGER
        recordings.to_file(f"{out_dir}/recordings.jsonl.gz")
        supervisions.to_file(f"{out_dir}/supervisions.jsonl.gz")
        cuts_raw.to_eager().to_file(f"{out_dir}/cuts.jsonl.gz")
        cuts_trim.to_file(f"{out_dir}/cuts_trim.jsonl.gz")  # <-- sẽ dùng file này cho bước tính feats

    def compute_and_store_fbank(self):
        out_dir = self.output_dir

        # ĐỌC EAGER (không dùng lazy ở bước tiền xử lý)
        cuts = load_manifest(f"{out_dir}/cuts_trim.jsonl.gz")

        # Tùy dataset bạn, chọn sampling_rate đúng với dữ liệu; nếu file gốc 16k thì để 16k
        fbank = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80, device="cuda"))

        # Tính & lưu feats (hàm này có thể tạo pipeline lazy tạm thời -> ép eager sau khi xong)
        cuts_with_feats = cuts.compute_and_store_features(
            extractor=fbank,
            storage_path="feats",   # dir chứa .llc
            num_jobs=1
        )

        # ÉP EAGER + LƯU: manifest cuối cùng dành cho TRAIN
        cuts_with_feats = cuts_with_feats.to_eager()
        cuts_with_feats.to_file(f"{out_dir}/cuts_with_feats_trim.jsonl.gz")

        # (Không bắt buộc) Nếu bạn cũng muốn phiên bản không trim:
        # cuts_raw = load_manifest(f"{out_dir}/cuts.jsonl.gz")
        # cuts_raw_feats = cuts_raw.compute_and_store_features(
        #     extractor=fbank, storage_path=f"{out_dir}/feats_raw", num_jobs=8
        # ).to_eager()
        # cuts_raw_feats.to_file(f"{out_dir}/cuts_with_feats.jsonl.gz")

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
def process():
    DATA_DIR = ""
    TXT_PATH = ""
    OUTPUT_DIR = "/root/S2T/data/manifests/"
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        list_audios = []
        list_transcripts = []
        for line in tqdm(lines):
            audio_path, text = line.split("\t")
            assert os.path.exists(audio_path), "File not found: {0}".format(audio_path)
            list_audios.append(audio_path)
            list_transcripts.append(text)
    print("[INFO] Total audios: {0}".format(len(list_audios))) 
    DataPreparation(list_audios, list_transcripts, OUTPUT_DIR, extract_fbank=False)

# if __name__ == "__main__":
#     process()
    # main()
    # SAMPLE: python prepare_data.py --data_path='/data/audio_data/valid.csv' --output_dir='/data/audio_data/valid_feats'