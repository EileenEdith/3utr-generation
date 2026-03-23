"""Legacy compatibility shim for GEMORNA shared-library imports.

The upstream `libg2m.so` binary expects a top-level `tokenization` module from
GEMORNA's original repository structure. This lightweight shim preserves that
import path for generation compatibility.
"""

init_token = '<sos>'
eos_token = '<eos>'


def seq_to_3mer_tokens(seq: str):
    seq = seq.upper().replace('T', 'U').replace(' ', '')
    tokens = []
    for i in range(0, len(seq), 3):
        chunk = seq[i:i+3]
        if len(chunk) < 3:
            chunk = chunk + ('N' * (3 - len(chunk)))
        tokens.append(chunk)
    return tokens


def tokenize_seq(text):
    tokens = []
    for token in text.split():
        if token in [init_token, eos_token] or (token.startswith('<') and token.endswith('>')):
            tokens.append(token)
        else:
            if len(token) == 3:
                tokens.append(token)
            else:
                tokens.extend(seq_to_3mer_tokens(token))
    return tokens


def tokenize_aa(protein):
    return list(protein)


def numericalize(text, vocab, sos_token=init_token, eos_token=eos_token):
    tokens = [sos_token] + tokenize_seq(text) + [eos_token]
    return [vocab[token] if token in vocab else vocab.get('<unk>', 0) for token in tokens]
