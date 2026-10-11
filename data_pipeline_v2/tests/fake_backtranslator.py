"""TEST-ONLY dummy back-translation backend (does no translation: tags the text). Never used on real data."""
def translate(texts, src, tgt):
    return [f"{t} <{src}->{tgt}>" for t in texts]
