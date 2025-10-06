import sentencepiece as spm

SPM_MODEL = "bpe_vi_50k.model"
TOK_OUT   = "tokens.txt"

sp = spm.SentencePieceProcessor()
sp.load(SPM_MODEL)

with open(TOK_OUT, "w", encoding="utf-8") as oh:
    # Reserve IDs for blank + unk
    oh.write("<blk> 0\n")
    oh.write("<unk> 1\n")

    next_id = 2
    for sp_id in range(sp.vocab_size()):
        piece = sp.id_to_piece(sp_id)
        if piece == "<unk>":     # already assigned to ID 1
            continue
        # SentencePiece uses "▁" to mark word-boundary/space; keep it as-is.
        oh.write(f"{piece} {next_id}\n")
        next_id += 1

print("Wrote", TOK_OUT)
