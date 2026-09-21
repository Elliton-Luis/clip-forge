"""peaks.py — seleção pelo auge (MODO EXPERIMENTAL, opt-in).

Ideia: em vez de ranquear janelas de duração quase-máxima pelo score médio,
detecta-se o PEAK (trecho de maior interesse) de cada candidato e constrói-se
o clip ao redor dele:

    context_start  peak_start  peak_end  context_end
    |              |           |         |
    └──── contexto ┤━━━ AUGE ━━├ contexto┘

Como o peak é identificado (sem fórmula arbitrária):
- `llm` (preferido): o LLM avalia SUBJANELAS pequenas (~12 s) do candidato —
  nunca o vídeo inteiro — e escolhe a de maior intensidade + motivo + se o
  auge precisa de contexto anterior. Mesma política de retry/pacing do scoring
  (reuso de scoring._call_with_retry-gates), 1 chamada por candidato.
- `heuristic` (fallback sem rede): ordenação lexicográfica de sinais JÁ
  observados pelo projeto — 1º sobreposição com risadas marcadas, 2º
  sobreposição com eventos acústicos, 3º densidade de exclamações (!/?),
  4º speech_rate local. Cada critério é individualmente significativo e a
  ordem está documentada aqui; `peak_source` deixa explícito qual caminho
  decidiu. Sem risadas/eventos no material, cai nos critérios textuais.
- `none`: peak nunca detectado (modo classic ou falha total).

Duração: o clip cresce do peak até cobrir frases inteiras e o mínimo, nunca
passa do máximo, nunca é estofado até o máximo. Diversidade: NMS sobre
OVERLAP DE PEAKS (>= 0.5 = mesmo momento, descarta) + teto por 10 min.

Não toca: legendas, alignment, scoring do modo classic.
"""
import re
import time
from dataclasses import dataclass

from .config import DEFAULT_MIN_SCORE, DEFAULT_MAX_PER_10MIN, NMS_DECAY_STRENGTH
from .models import Candidate

SELECTION_MODES = ("classic", "peak")
PEAK_WINDOW_SEC = 12.0  # subjanela de julgamento (pequena de propósito)
PEAK_STEP_SEC = 6.0
PEAK_SAME_MOMENT = 0.5  # overlap de peaks >= 50% = mesmo momento
PEAK_VERSION = 1

PEAK_JUDGE_PROMPT = """Você é um editor de cortes virais. Você recebe SUBJANELAS de ~12 segundos de UM candidato a clip (transcrição + tempos + energia da janela toda + speech_rate local).

Para CADA subjanela, dê "intensity" 0-10: quanto vale a pena MOSTRAR exatamente este trecho (punchline, reação, reviravolta, frase de efeito, interação engraçada — não exposição genérica).
Escolha UMA "peak": a subjanela de maior intensidade (o auge). Dê "peak_reason" curto (o que acontece no auge, só com palavras do texto) e "needs_context" (true se o auge só faz sentido com o que vem ANTES).

Responda SOMENTE JSON: {"windows": [{"id": 0, "intensity": 7.5}], "peak": 0, "peak_reason": "motivo", "needs_context": true}
Avalie TODAS as subjanelas, na ordem, pelos ids."""


@dataclass
class SubWindow:
    idx: int
    start: float
    end: float
    text: str
    words: list


def resolve_selection_mode(name: str | None) -> str:
    import os as _os
    raw = (name if name is not None
           else _os.environ.get("CLIPPER_SELECTION_MODE", "classic")).lower()
    if raw not in SELECTION_MODES:
        raise ValueError(
            f"selection mode '{raw}' desconhecido — opções: {list(SELECTION_MODES)}")
    return raw


def split_subwindows(c: Candidate, window: float = PEAK_WINDOW_SEC,
                     step: float = PEAK_STEP_SEC) -> list[SubWindow]:
    """Candidato → subjanelas de ~window s com passo step, snap em fronteira
    de palavra. 1 janela se o candidato for curto."""
    words = sorted(c.words, key=lambda w: w.start)
    if not words:
        return [SubWindow(0, c.start, c.end, c.text, [])]
    out: list[SubWindow] = []
    t = c.start
    idx = 0
    while t < c.end - 0.5:
        tend = min(t + window, c.end)
        ww = [w for w in words if w.end > t and w.start < tend]
        if ww:
            s = min(w.start for w in ww)
            e = max(w.end for w in ww)
            out.append(SubWindow(idx, s, e,
                                 " ".join(w.text.strip() for w in ww
                                          if w.text and w.text.strip()), ww))
            idx += 1
        if tend >= c.end:
            break
        t += step
    return out or [SubWindow(0, c.start, c.end, c.text, list(words))]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _signal_row(s: SubWindow, laughs: list, events: list) -> tuple:
    """Sinais observados da subjanela (para ordenação lexicográfica)."""
    laugh_s = sum(_overlap(s.start, s.end, getattr(e, "start", 0), getattr(e, "end", 0))
                  for e in (laughs or []))
    event_s = sum(_overlap(s.start, s.end, getattr(e, "start", 0), getattr(e, "end", 0))
                  for e in (events or []))
    excl = len(re.findall(r"[!?…?]+", s.text))
    n = max(1, len(s.words))
    dur = max(0.5, s.end - s.start)
    return (round(laugh_s, 2), round(event_s, 2), excl / n, round(n / dur, 2))


def detect_peak_heuristic(c: Candidate, laughs: list | None = None,
                          events: list | None = None) -> Candidate:
    """Fallback sem rede: melhor subjanela pela ordem
    risada > evento acústico > exclamação > fala rápida.
    peak_score neutro (5.0): a heurística LOCALIZA o auge, não julga
    intensidade — o desempate em select_peak cai no score da janela.
    `peak_source` registra o caminho honestamente."""
    subs = split_subwindows(c)
    if len(subs) == 1:
        s = subs[0]
        _set_peak(c, s.start, s.end, 5.0, "heuristic", "janela única")
        return c
    ranked = sorted(range(len(subs)),
                    key=lambda i: (_signal_row(subs[i], laughs, events),
                                   -subs[i].start),
                    reverse=True)
    # Desempate estável: sinais iguais → subjanela mais cedo primeiro.
    best = subs[ranked[0]]
    row = _signal_row(best, laughs, events)
    _set_peak(c, best.start, best.end, 5.0, "heuristic",
              f"risada={row[0]}s evento={row[1]}s excl={row[2]:.2f} sr={row[3]:.1f}")
    c._sub_rank = ranked  # type: ignore
    c._subs = subs  # type: ignore
    return c


def _set_peak(c: Candidate, start: float, end: float, score: float,
              source: str, reason: str) -> None:
    if c.window_start is None:
        c.window_start, c.window_end = c.start, c.end
    c.peak_start, c.peak_end = start, max(end, start + 0.5)
    c.peak_score = max(0.0, min(10.0, score))
    c.peak_source = source
    c.peak_reason = reason[:200]


def judge_peaks_llm(candidates: list, model: str,
                    context: str | None = None) -> dict:
    """1 chamada LLM por candidato com suas subjanelas. Retorna stats.
    Falha por candidato → fallback heurístico (peak_source registra)."""
    from .scoring import _load_api_keys, _KeyGate, _build_prompt
    from openai import OpenAI
    from .config import NVIDIA_BASE_URL
    keys = _load_api_keys()
    clients = [OpenAI(base_url=NVIDIA_BASE_URL, api_key=k,
                      timeout=180, max_retries=0) for k in keys]
    gates = [_KeyGate() for _ in clients]
    prompt = _build_prompt(PEAK_JUDGE_PROMPT, context, None)
    stats = {"requests": 0, "llm": 0, "heuristic": 0}
    for c in candidates:
        subs = split_subwindows(c)
        c._subs = subs  # type: ignore
        if len(subs) == 1:
            _set_peak(c, subs[0].start, subs[0].end, 5.0, "heuristic", "janela única")
            stats["heuristic"] += 1
            continue
        id_map = {s.idx: s for s in subs}
        try:
            result = _call_peak(clients, model, c, subs, prompt, gates)
            stats["requests"] += 1
            _apply_peak_result(c, subs, id_map, result)
            stats["llm" if c.peak_source == "llm" else "heuristic"] += 1
        except Exception as e:
            print(f"      ! peak LLM falhou ({c.start:.0f}-{c.end:.0f}s: {e}) — heurística")
            detect_peak_heuristic(c)
            stats["heuristic"] += 1
    return stats


def _call_peak(clients, model: str, c: Candidate, subs: list,
               prompt: str, gates) -> dict:
    import json as _json
    import time as _t

    def _payload():
        return {"candidate": {"start": round(c.start, 1), "end": round(c.end, 1),
                              "energy": c.energy},
                "windows": [{"id": s.idx, "transcript": s.text[:600],
                             "start": round(s.start, 1), "end": round(s.end, 1),
                             "speech_rate": round(len(s.words) / max(0.5, s.end - s.start), 2)}
                            for s in subs]}

    # Retry próprio (3 tentativas, backoff 10/30 s) com o mesmo pacing por
    # chave do scoring; payload específico do julgamento.
    import time as _t
    last = None
    for attempt in range(3):
        client = clients[attempt % len(clients)]
        gates[attempt % len(clients)].wait()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": _json.dumps(
                              _payload(), ensure_ascii=False)}],
                temperature=0.3, max_tokens=2000)
            import re as _re
            content = (resp.choices[0].message.content or "").strip()
            content = _re.sub(r"<think>.*?</think>", "", content,
                              flags=_re.DOTALL | _re.IGNORECASE)
            m = _re.search(r"\{.*\}", content, flags=_re.DOTALL)
            return _json.loads(m.group(0) if m else content)
        except Exception as e:
            last = e
            print(f"      ! peak judge tentativa {attempt+1}/3: {e}")
            if attempt < 2:
                _t.sleep([10, 30][attempt])
    raise last  # type: ignore


def _apply_peak_result(c: Candidate, subs: list, id_map: dict, result: dict) -> None:
    """Aplica o julgamento; ids inválidos ou peak fora → heurística (nunca chuta)."""
    inten = {}
    for item in result.get("windows", []):
        try:
            i = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if i in id_map:
            try:
                inten[i] = max(0.0, min(10.0, float(item.get("intensity", 0))))
            except (TypeError, ValueError):
                continue
    try:
        peak_id = int(result.get("peak"))
    except (TypeError, ValueError):
        peak_id = -1
    if peak_id not in id_map or peak_id not in inten:
        print(f"      ! peak LLM inválido ({c.start:.0f}s) — heurística")
        detect_peak_heuristic(c)
        return
    c._sub_scores = inten  # type: ignore
    s = id_map[peak_id]
    reason = str(result.get("peak_reason", ""))[:200]
    _set_peak(c, s.start, s.end, inten[peak_id], "llm", reason)
    c._needs_context = bool(result.get("needs_context", True))  # type: ignore


def build_clip_around_peak(c: Candidate, min_dur: float, max_dur: float,
                           media_end: float | None = None) -> Candidate:
    """Expande do peak até frases inteiras + mínimo; nunca estofa até o máximo.
    Preserva window_* e escreve start/end finais (= context_start/end)."""
    if c.peak_start is None or c.peak_end is None:
        detect_peak_heuristic(c)
    if c.window_start is None:
        c.window_start, c.window_end = c.start, c.end
    words = sorted(c.words, key=lambda w: w.start)
    lo_bound = c.window_start if c.window_start is not None else c.start
    hi_bound = c.window_end if c.window_end is not None else c.end
    if media_end is not None:
        hi_bound = min(hi_bound, media_end)

    def _sent_expand(s: float, e: float) -> tuple[float, float]:
        # Frase = até . ! ? … (unidade de compreensão; sem ela o peak perde sentido).
        starts = [w.start for w in words if w.end <= s + 0.01]
        ends = [w.end for w in words if w.start >= e - 0.01]
        s2 = s
        for w in reversed(words):
            if w.end <= s + 0.01 and not re.search(r"[.!?…]\s*$", w.text):
                s2 = w.start
            else:
                break
        e2 = e
        for w in words:
            if w.start >= e - 0.01:
                e2 = w.end
                if re.search(r"[.!?…]\s*$", w.text):
                    break
            elif w.end > e:
                e2 = max(e2, w.end)
        return (max(s2, lo_bound), min(e2, hi_bound))

    s, e = _sent_expand(c.peak_start, c.peak_end)
    subs = getattr(c, "_subs", None)
    scores = getattr(c, "_sub_scores", None) or {}

    def _side_score(side: str) -> float:
        if not subs:
            return 0.0
        if side == "before":
            cand = [i for i, x in enumerate(subs) if x.end <= s + 0.01]
            key = max(cand) if cand else None
        else:
            cand = [i for i, x in enumerate(subs) if x.start >= e - 0.01]
            key = min(cand) if cand else None
        if key is None:
            return -1.0
        return scores.get(subs[key].idx, 0.0)

    guard = 0
    while e - s < min_dur and guard < 50:
        guard += 1
        b, a = _side_score("before"), _side_score("after")
        if b < 0 and a < 0:
            break
        if b >= a and s > lo_bound:
            s = max(lo_bound, s - PEAK_STEP_SEC)
            s = max([w.start for w in words if w.start <= s + 0.01] or [s])
            s = max(s, lo_bound)
        elif e < hi_bound:
            e = min(hi_bound, e + PEAK_STEP_SEC)
            e = min([w.end for w in words if w.end >= e - 0.01] or [e])
            e = min(e, hi_bound)
        else:
            break
    if e - s > max_dur:
        # Corta contexto (nunca o peak): primeiro o lado mais longe do peak.
        excess = (e - s) - max_dur
        cut_b = min(excess, max(0.0, s - lo_bound), max(0.0, c.peak_start - s))
        s += cut_b
        excess -= cut_b
        if excess > 0:
            e -= min(excess, max(0.0, e - (c.peak_end or e)))
        if e - s > max_dur:  # peak maior que max: centra o max no meio do peak
            mid = (c.peak_start + c.peak_end) / 2
            s, e = mid - max_dur / 2, mid + max_dur / 2
            s = max([w.start for w in words if w.start <= s] or [s])
            e = min([w.end for w in words if w.end >= e] or [e])
    s, e = max(0.0, s), min(e, hi_bound)
    c.start, c.end = s, e
    dur = max(0.5, e - s)
    c.speech_rate = round(len(words) / dur, 2)
    return c


def peak_overlap_ratio(a: Candidate, b: Candidate) -> float:
    """Overlap dos PEAKS / menor peak. Sem peak em algum lado → 0 (não suprime)."""
    if a.peak_start is None or b.peak_start is None:
        return 0.0
    a1 = a.peak_end if a.peak_end is not None else a.peak_start
    b1 = b.peak_end if b.peak_end is not None else b.peak_start
    over = max(0.0, min(a1, b1) - max(a.peak_start, b.peak_start))
    shortest = min(a1 - a.peak_start, b1 - b.peak_start)
    return (over / shortest) if shortest > 0 else 0.0


def select_peak(candidates: list, top_n: int,
                min_score: float = DEFAULT_MIN_SCORE,
                max_per_10min: int = DEFAULT_MAX_PER_10MIN) -> list:
    """Ranqueia por peak_score (intensidade do auge), filtra pelo score da
    janela (porta de qualidade já calibrada) e suprime mesmo-momento por
    overlap de PEAK — clips 10-70/20-80/30-90 do mesmo auge não coexistem."""
    pool = [c for c in candidates if not c.failed and c.score >= min_score]
    if not pool:
        print(f"   ! nenhum candidato >= {min_score} (de {len(candidates)}). Tente --min-score menor.")
        return []
    for c in pool:
        if c.peak_start is None:
            detect_peak_heuristic(c)
    pool.sort(key=lambda c: (c.peak_score, c.score), reverse=True)
    selected: list = []
    buckets: dict[int, int] = {}
    for c in pool:
        bucket = int(c.start // 600)
        if buckets.get(bucket, 0) >= max_per_10min:
            continue
        if any(peak_overlap_ratio(c, s) >= PEAK_SAME_MOMENT for s in selected):
            continue  # mesmo momento já coberto por peak melhor
        max_overlap = max((c.overlap_ratio(s) for s in selected), default=0.0)
        penalized = c.score * (1 - NMS_DECAY_STRENGTH * max_overlap) if selected else c.score
        if penalized < min_score:
            continue
        c._penalized_score = penalized  # type: ignore
        selected.append(c)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if len(selected) >= top_n:
            break
    selected.sort(key=lambda c: c.start)
    print(f"[4/5-peak] {len(selected)} clipes pelo auge "
          f"(de {len(pool)} elegíveis, min_score={min_score}, max_per_10min={max_per_10min})")
    if selected:
        print("      peaks: " + ", ".join(
            f"{c.peak_score:.1f}[{c.peak_start:.0f}-{c.peak_end:.0f}]/{c.peak_source}" for c in selected))
    return selected


PEAK_TITLE_PROMPT = """Você recebe CLIPS, cada um com o TRANSCRITO COMPLETO do candidato e o AUGE (trecho de maior interesse, com tempos). Para CADA clip, escreva um "title" de até 50 caracteres, sem emoji, que represente O AUGE ou o acontecimento central do clip.

Regra absoluta: use SOMENTE palavras, pessoas, fatos e acontecimentos presentes no texto. Nunca adicione informação nova. Na dúvida entre chamativo e fiel, escolha fiel.
Responda SOMENTE JSON, no formato: {"clips": [{"id": 0, "title": "..."}, ...]} — um item por clip recebido, na ordem, usando o "id" fornecido."""


def _parse_peak_titles(content: str):
    """Parse defensivo: {"clips":[...]} | [...] | objetos concatenados."""
    import json as _json
    import re as _re
    m = _re.search(r"\{.*\}", content, flags=_re.DOTALL)
    try:
        return _json.loads(m.group(0) if m else content)
    except Exception:
        pass
    items = []
    for m in _re.finditer(r"\{[^{}]*\}", content):
        try:
            items.append(_json.loads(m.group(0)))
        except Exception:
            continue
    return items


def retitle_peak(selected: list, model: str) -> dict:
    """Títulos do auge só p/ selecionados (1 chamada batelada). Falha → mantém
    o título da janela (title_source fica "window")."""
    import json as _json
    import re as _re
    from openai import OpenAI
    from .config import NVIDIA_BASE_URL
    from .scoring import _load_api_keys
    from .review import validate_grounding
    stats = {"retitled": 0, "kept": 0}
    if not selected:
        return stats
    try:
        keys = _load_api_keys()
    except SystemExit:
        print("   ! peak titles: sem chave API — mantém títulos da janela")
        stats["kept"] = len(selected)
        return stats
    approved = " ".join(c.text for c in selected)
    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=keys[0],
                        timeout=180, max_retries=0)
        payload = {"clips": [
            {"id": i, "transcript": c.text[:1500],
             "peak": " ".join(w.text.strip() for w in c.words
                              if c.peak_start is not None and c.peak_end is not None
                              and w.end > c.peak_start and w.start < c.peak_end)[:600],
             "peak_start": round(c.peak_start or 0, 1),
             "peak_end": round(c.peak_end or 0, 1)} for i, c in enumerate(selected)]}
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": PEAK_TITLE_PROMPT},
                                   {"role": "user", "content": _json.dumps(
                                       payload, ensure_ascii=False)}],
            temperature=0.3, max_tokens=2000)
        content = (resp.choices[0].message.content or "").strip()
        content = _re.sub(r"<think>.*?</think>", "", content,
                          flags=_re.DOTALL | _re.IGNORECASE)
        data = _parse_peak_titles(content)
        items = data if isinstance(data, list) else data.get("clips", [])
        by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
        for i, c in enumerate(selected):
            t = str((by_id.get(i) or {}).get("title", "") or "")[:50].strip()
            if not t:
                stats["kept"] += 1
                continue
            bad = validate_grounding(t, approved + " " + c.text)
            if bad:
                print(f"   ! peak title clipe {i+1} fora do transcript {bad} — mantém janela")
                stats["kept"] += 1
                continue
            c.title = t
            c.title_source = "peak"
            stats["retitled"] += 1
    except Exception as e:
        print(f"   ! peak titles falhou ({e}) — mantém títulos da janela")
        stats["kept"] += len(selected) - stats["retitled"]
    return stats


def detect_all(candidates: list, model: str | None,
               context: str | None = None, use_llm: bool = True,
               laughs: list | None = None, events: list | None = None) -> dict:
    """Pipeline da etapa peak p/ todos os candidatos: julga (LLM ou heurística)
    e constrói o clip ao redor do auge. Retorna stats."""
    t0 = time.time()
    if use_llm and model:
        stats = judge_peaks_llm([c for c in candidates if not c.failed], model, context)
    else:
        stats = {"requests": 0, "llm": 0, "heuristic": 0}
        for c in candidates:
            if not c.failed and c.peak_start is None:
                detect_peak_heuristic(c, laughs, events)
                stats["heuristic"] += 1
    return {**stats, "time_sec": round(time.time() - t0, 3)}
