import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="bpe_vi_50k",
    model_type="bpe",
    vocab_size=50000,
    character_coverage=1.0,
    unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
    input_sentence_size=2_000_000,
    shuffle_input_sentence=True,
    byte_fallback=False,
    hard_vocab_limit=False,
    num_threads=8
)
