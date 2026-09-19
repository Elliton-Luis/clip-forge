"""quality.py — detecção de alucinação por repetição (só texto, sem modelo).
SRP: dizer se um trecho transcrito tem assinatura de loop alucinatório.
NÃO mede tempo, NÃO reescreve timestamp, NÃO decide default — só sinaliza.

Assinaturas (calibradas em material real, 2026-09):
- token único repetido em sequência >= 10 ("♪" x17, "plataforma" em loop).
  Fala real ("dá, dá, dá, dá" = 4) passa longe do limiar.
- n-grama de >= 4 palavras ocorrendo >= 3 vezes ("toda a plataforma" x5).

Saída: {"flagged": bool, "reason": str, "detail": str}. Conservador por
desenho: prefere falso-negativo a acusar fala real enfática.
"""

SINGLE_RUN_MIN = 10
NGRAM_MIN_N = 4
NGRAM_MIN_COUNT = 3


def _tokens(text: str) -> list[str]:
    import string
    strip = string.punctuation + ".,!?;:…—–«»“”‘’´`´"
    out = []
    for raw in (text or "").split():
        t = raw.strip(strip).lower()
        if t:
            out.append(t)
    return out


def single_run(text: str) -> tuple[str, int] | None:
    """Retorna (token, tamanho) da maior sequência idêntica, se >= limiar."""
    toks = _tokens(text)
    best: tuple[str, int] | None = None
    run_tok, run_len = None, 0
    for t in toks + [None]:
        if t == run_tok:
            run_len += 1
        else:
            if run_tok is not None and run_len >= SINGLE_RUN_MIN:
                if best is None or run_len > best[1]:
                    best = (run_tok, run_len)
            run_tok, run_len = t, 1
    return best


def repeated_ngram(text: str) -> tuple[str, int] | None:
    """Retorna (ngrama, ocorrências) do maior n>=4 com >=3 ocorrências."""
    toks = _tokens(text)
    n = len(toks)
    for size in range(min(12, n // NGRAM_MIN_COUNT), NGRAM_MIN_N - 1, -1):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(n - size + 1):
            g = tuple(toks[i:i + size])
            counts[g] = counts.get(g, 0) + 1
        cands = [(g, c) for g, c in counts.items() if c >= NGRAM_MIN_COUNT]
        if cands:
            g, c = max(cands, key=lambda kv: (len(kv[0]), kv[1]))
            return (" ".join(g), c)
    return None


def check(text: str) -> dict:
    """Sinaliza loop alucinatório. Puro e testável."""
    run = single_run(text)
    if run:
        return {"flagged": True, "reason": "single_token_run",
                "detail": f"{run[0]!r} x{run[1]}"}
    ng = repeated_ngram(text)
    if ng:
        return {"flagged": True, "reason": "repeated_ngram",
                "detail": f"{ng[0]!r} x{ng[1]}"}
    return {"flagged": False, "reason": "ok", "detail": ""}


def choose_retry(original_flagged: bool, retry_flagged: bool) -> str:
    """Qual transcrição manter. Regra documentada: só troca se o retry LIMPA
    a sinalização; se ambos limpos, original (determinismo); se ambos sujos,
    original (não troca alucinação por alucinação)."""
    if original_flagged and not retry_flagged:
        return "retry"
    return "original"
