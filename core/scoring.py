"""scoring.py — prompt, chamada NIM e pontuação em lotes.

SRP: só scoring. OCP: prompt extensível via context/examples sem editar código.
"""
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .config import NVIDIA_BASE_URL, DEFAULT_MODEL, SEGMENTS_PER_SCORING_CALL
from .cache import load_scores, save_scores

SYSTEM_PROMPT = """Você é um editor especialista em cortes virais de lives e \
gameplay para TikTok/Reels/Shorts. Você recebe trechos transcritos de uma live ou \
sessão de jogo e precisa avaliar o POTENCIAL VIRAL de cada um.

Dê nota alta para: jogadas insanas ou clutch, reações de hype genuínas, momentos \
engraçados/memeáveis, falas polêmicas ou opiniões fortes, treta ou banter com chat/duo, \
rage ou fail cômico, reviravoltas (quase perdeu e virou / dominava e perdeu), \
frases de efeito que funcionam fora de contexto (autocontidas).

Dê nota baixa para: trechos genéricos, silêncio narrado, explicações técnicas sem \
graça, momentos que só fazem sentido com contexto anterior extenso.

Responda SOMENTE com um JSON válido (sem markdown, sem texto antes ou depois), no formato:
{"clips": [{"id": 0, "score": 8.5, "reason": "motivo curto", "title": "título curto e chamativo até 50 caracteres", "hashtags": "#tag1 #tag2 #tag3"}, ...]}

Regras obrigatórias:
- A nota "score" vai de 0 a 10.
- "title" deve ter no máximo 50 caracteres, sem emoji.
- O "title" é um recorte do transcript: use SOMENTE palavras, pessoas, fatos e
  acontecimentos presentes no texto transcrito. Nunca adicione informação nova
  (ex.: se o transcript não menciona um grupo, característica ou evento, o
  título não pode mencioná-los). Na dúvida entre chamativo e fiel, escolha fiel.
- "hashtags" deve conter exatamente 3 hashtags em minúsculas.
- Avalie TODOS os clipes recebidos, na ordem, usando o "id" fornecido.
- Use também a duração, energia de áudio (alta/media/baixa) e speech_rate como pistas: energia alta + fala rápida costuma indicar hype/grito.
"""


def _build_prompt(base: str, context: str | None, examples: list | None) -> str:
    parts = [base]
    if context:
        parts.append(f"\nContexto da live/sessão: {context}")
    if examples:
        parts.append("\nExemplos de referência (few-shot):")
        for ex in examples[:4]:
            parts.append(
                f"- transcript: \"{ex.get('transcript','')[:300]}\" => "
                f"score {ex.get('score','?')}, reason: \"{ex.get('reason','')}\", "
                f"title: \"{ex.get('title','')}\" hashtags: {ex.get('hashtags','')}"
            )
        parts.append("Use esses exemplos como calibração.")
    return "\n".join(parts)


def _load_examples(path: str | None) -> list:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"   ! examples não encontrado: {p} — ignorando")
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "examples" in data:
            data = data["examples"]
        if not isinstance(data, list):
            raise ValueError("examples.json deve ser lista ou {\"examples\": [...]}")
        valid = [ex for ex in data[:4] if isinstance(ex, dict) and "transcript" in ex]
        print(f"   -> {len(valid)} few-shot de {p}")
        return valid
    except Exception as e:
        print(f"   ! falha ao carregar examples ({p}): {e} — ignorando")
        return []


def _strip_reasoning(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    return m.group(0) if m else content


def _clamp(v) -> float:
    try:
        f = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(10.0, f))


def _sanitize_hashtags(raw: str) -> str:
    tags = re.findall(r"#\w+", raw or "")
    tags = [t.lower() for t in tags]
    while len(tags) < 3:
        tags.append("#clip")
    return " ".join(tags[:3])


def _call_nim(client, model: str, batch: list, prompt: str) -> tuple[dict, dict | None]:
    payload = {
        "clips": [
            {"id": idx, "transcript": c.text[:1500],
             "start": round(c.start, 1), "end": round(c.end, 1),
             "duration": round(c.duration, 1), "energy": c.energy, "speech_rate": c.speech_rate}
            for idx, c in batch
        ]
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.3, max_tokens=4000,
    )
    usage = getattr(resp, "usage", None)
    tokens = None
    if usage is not None:
        try:
            tokens = {
                "prompt": getattr(usage, "prompt_tokens", None),
                "completion": getattr(usage, "completion_tokens", None),
                "total": getattr(usage, "total_tokens", None),
            }
        except Exception:
            tokens = None
    content = _strip_reasoning((resp.choices[0].message.content or "").strip())
    return json.loads(content), tokens


def _configured_keys() -> list[str]:
    """Chaves presentes no env, sem exigir. Puro quanto à ordem."""
    raw = os.environ.get("NVIDIA_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        single = (os.environ.get("NVIDIA_API_KEY") or "").strip()
        if single:
            keys = [single]
    return keys


def _load_api_keys() -> list[str]:
    """Chaves NVIDIA (rodízio por lote/tentativa). Puro quanto à ordem.

    `NVIDIA_API_KEYS` (vírgula) tem prioridade; senão `NVIDIA_API_KEY` única.
    Valores nunca são logados — só o índice (chave 1/2). Sem chave: erro claro.
    """
    keys = _configured_keys()
    if not keys:
        sys.exit("Erro: defina NVIDIA_API_KEY ou NVIDIA_API_KEYS "
                 "(https://build.nvidia.com).")
    return keys


_key_cursor = 0  # rodízio global entre chaves (resetável em testes)


# Concorrência limitada (§7): 1 worker por chave, teto de 3. Requests de
# ~27 s geram ~2 RPM/worker — longe do limite; o pacing abaixo é só rede
# de segurança (vale também para retries, que passam pelo mesmo caminho).
SCORING_MAX_WORKERS = 3
# 40 RPM por chave = 1 req/1,5 s; margem conservadora: 1 req/2 s (~30 RPM).
KEY_MIN_INTERVAL_SEC = 2.0
# Teto do Retry-After honrado em 429 (nunca dormir para sempre).
RATE_LIMIT_WAIT_CAP_SEC = 120.0


class _KeyGate:
    """Pacing por chave: intervalo mínimo entre inícios de request.

    Thread-safe; toda tentativa (incluindo retry) passa por aqui antes de
    tocar a API, então retries nunca furam o rate limit (§9).
    """

    def __init__(self, min_interval: float = KEY_MIN_INTERVAL_SEC):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            gap = now - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()


def _next_client(clients: list):
    """Próximo cliente em round-robin. Retorna (client, idx); valor nunca logado."""
    global _key_cursor
    idx = _key_cursor % len(clients)
    _key_cursor += 1
    return clients[idx], idx


def _rate_limit_wait(exc: Exception, default: float) -> float | None:
    """Segundos a aguardar se exc for 429, senão None.

    Honra `Retry-After` da API quando presente (limitado ao teto);
    sem ele, usa o backoff padrão. Detecção por `status_code` (SDK) ou
    pelo texto "429" (exceções genéricas) — nunca pelo tipo da exceção.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        try:
            status = 429 if "429" in str(exc) else None
        except Exception:
            status = None
    if status != 429:
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        ra = float(headers.get("retry-after", ""))
        return max(0.0, min(ra, RATE_LIMIT_WAIT_CAP_SEC))
    except (TypeError, ValueError):
        return default


def _bump(stats: dict | None, lock, key: str, delta: float = 1) -> None:
    if stats is None:
        return
    if lock is not None:
        with lock:
            stats[key] = stats.get(key, 0) + delta
    else:
        stats[key] = stats.get(key, 0) + delta


def _note_latency(stats: dict | None, lock, seconds: float) -> None:
    """Soma/min/max de latência por tentativa (base do avg e do speedup)."""
    if stats is None:
        return

    def _add():
        stats["total_time"] = stats.get("total_time", 0.0) + seconds
        cur_min = stats.get("min_lat")
        stats["min_lat"] = seconds if cur_min is None else min(cur_min, seconds)
        cur_max = stats.get("max_lat")
        stats["max_lat"] = seconds if cur_max is None else max(cur_max, seconds)

    if lock is not None:
        with lock:
            _add()
    else:
        _add()


def _say(progress, text: str) -> None:
    if progress is not None:
        progress.note(text)
    else:
        print(text)


def _warn(progress, text: str) -> None:
    if progress is not None:
        progress.warn(text)
    else:
        print(text)


def _detail(progress, text: str) -> None:
    if progress is not None:
        progress.detail(text)
    else:
        print(text)


def _call_with_retry(clients, model: str, batch: list, prompt: str, retries: int = 3,
                     stats: dict | None = None,
                     key_hint: int | None = None,
                     gates: list | None = None,
                     lock=None, progress=None) -> dict:
    # clients: lista não-vazia de OpenAI. Caminho serial (key_hint None):
    # cada tentativa usa a próxima chave do rodízio global. Caminho paralelo:
    # o lote tem afinidade com a chave key_hint e só roda para a próxima em
    # retry — quota diluída, distribuição determinística (lote i → chave i%n).
    # Toda tentativa passa pelo gate da chave (pacing) antes da API.
    backoffs = [10, 30, 60]
    last = None
    nkeys = len(clients)
    for attempt in range(retries):
        kidx = ((key_hint + attempt) % nkeys) if key_hint is not None else None
        if kidx is None:
            client, kidx = _next_client(clients)
        else:
            client = clients[kidx]
        if gates is not None:
            gates[kidx].wait()
        _bump(stats, lock, "requests")
        t0 = time.time()
        try:
            result, tokens = _call_nim(client, model, batch, prompt)
            _bump(stats, lock, "successes")
            _note_latency(stats, lock, time.time() - t0)
            if tokens:
                if lock is not None:
                    with lock:
                        for k_src, k_dst in (("prompt", "prompt_tokens"),
                                             ("completion", "completion_tokens"),
                                             ("total", "total_tokens")):
                            v = tokens.get(k_src)
                            if v is not None:
                                stats[k_dst] = (stats.get(k_dst) or 0) + v
                else:
                    for k_src, k_dst in (("prompt", "prompt_tokens"),
                                         ("completion", "completion_tokens"),
                                         ("total", "total_tokens")):
                        v = tokens.get(k_src)
                        if v is not None:
                            stats[k_dst] = (stats.get(k_dst) or 0) + v
            return result
        except json.JSONDecodeError as e:
            last = e
            _warn(progress,
                  f"      ! JSON inválido (tentativa {attempt+1}/{retries}, "
                  f"chave {kidx+1}/{nkeys}): {e}")
        except Exception as e:
            last = e
            _warn(progress,
                  f"      ! Erro NIM (tentativa {attempt+1}/{retries}, "
                  f"chave {kidx+1}/{nkeys}): {e}")
        _note_latency(stats, lock, time.time() - t0)
        if attempt == retries - 1:
            _bump(stats, lock, "failures")
        else:
            _bump(stats, lock, "retries")
        if attempt < retries - 1:
            wait = backoffs[min(attempt, len(backoffs) - 1)]
            rl_wait = _rate_limit_wait(last, wait)
            if rl_wait is not None:
                _bump(stats, lock, "rate_limited")
                if rl_wait != wait:
                    _detail(progress,
                            f"        429 na chave {kidx+1}/{nkeys}: aguardando "
                            f"{rl_wait:.0f}s (Retry-After)...")
                else:
                    _detail(progress,
                            f"        429 na chave {kidx+1}/{nkeys}: aguardando {wait}s...")
                time.sleep(rl_wait)
            else:
                _detail(progress, f"        aguardando {wait}s...")
                time.sleep(wait)
    raise last  # type: ignore


def _apply_batch_result(batch: list, id_map: dict, result: dict,
                        progress=None) -> None:
    """Associa respostas aos candidatos pelos ids (nunca por ordem de
    conclusão). Id ausente ou fora do lote = candidato `failed`."""
    ids = {idx for idx, _ in batch}
    seen: set[int] = set()
    for item in result.get("clips", []):
        idx = item.get("id")
        if idx is None or idx not in ids:
            continue
        seen.add(idx)
        c = id_map[idx]
        c.score = _clamp(item.get("score", 0))
        c.reason = str(item.get("reason", ""))[:200]
        c.title = str(item.get("title", ""))[:50]
        c.hashtags = _sanitize_hashtags(item.get("hashtags", ""))
        c.failed = False
    for idx, c in batch:
        if idx not in seen:
            c.failed = True
            _warn(progress, f"      ! candidato {idx} sem retorno — sem nota")


def _fail_batch(batch: list, n: int, num: int, e: Exception, progress=None) -> None:
    _warn(progress, f"   ! Falha lote {num}/{n}: {e}. Marcando sem nota.")
    for _, c in batch:
        c.failed = True


def score(candidates: list, model: str = DEFAULT_MODEL,
          cache_dir: Path | None = None, fingerprint: str | None = None,
          force: bool = False, context: str | None = None, examples_path: str | None = None,
          metrics=None, progress=None) -> list:
    if progress is not None:
        # Total conhecido antes de começar (lotes sobre candidatos).
        _nb = max(1, math.ceil(len(candidates) / SEGMENTS_PER_SCORING_CALL))
        progress.stage("scoring", "Scoring", total=_nb, unit="lotes")
    if cache_dir and fingerprint and not force:
        if load_scores(cache_dir, fingerprint, candidates, progress):
            if progress is not None:
                progress.skip("scoring", "cache")
            if metrics is not None:
                try:
                    metrics.set_nvidia(model=model, requests=0, successes=0,
                                       failures=0, retries=0, total_time_sec=0.0,
                                       avg_latency_sec=None,
                                       keys=len(_configured_keys()) or 1)
                except Exception:
                    pass
            return candidates

    from openai import OpenAI
    keys = _load_api_keys()

    # Auditoria (API): timeout explícito + max_retries=0 no client. Sem isso,
    # cada chamada poderia travar até 600s (default) e ainda sofrer retries
    # internos do SDK além dos nossos 3× com backoff 10/30/60 — paralisando um
    # job de 1h40. A única política de retry é _call_with_retry (limitada).
    # Um client por chave: rodízio por lote e por tentativa (nunca loga valores).
    clients = [OpenAI(base_url=NVIDIA_BASE_URL, api_key=k,
                      timeout=180, max_retries=0) for k in keys]
    if progress is None and len(clients) > 1:
        print(f"   -> rodízio entre {len(clients)} chaves NVIDIA (resiliência a quota/503)")
    elif progress is not None:
        progress.detail(f"rodízio entre {len(clients)} chaves NVIDIA")
    examples = _load_examples(examples_path)
    prompt = _build_prompt(SYSTEM_PROMPT, context, examples)

    n_batches = math.ceil(len(candidates) / SEGMENTS_PER_SCORING_CALL)
    if progress is None:
        print(f"[3/5] Pontuando {len(candidates)} candidatos em {n_batches} chamada(s)...")
    id_map = {idx: c for idx, c in enumerate(candidates)}
    stats = {"requests": 0, "successes": 0, "failures": 0, "retries": 0,
             "rate_limited": 0,
             "total_time": 0.0, "min_lat": None, "max_lat": None,
             "prompt_tokens": 0, "completion_tokens": 0,
             "total_tokens": 0}
    indexed = list(enumerate(candidates))
    batches = [indexed[s:s + SEGMENTS_PER_SCORING_CALL]
               for s in range(0, len(candidates), SEGMENTS_PER_SCORING_CALL)]

    # Concorrência limitada: 1 worker por chave (teto 3). Com 1 chave o
    # caminho é idêntico ao serial anterior — sem regressão possível.
    workers = min(len(clients), SCORING_MAX_WORKERS)
    gates = [_KeyGate(KEY_MIN_INTERVAL_SEC) for _ in clients]
    t_wall = time.time()
    if workers <= 1:
        for num, batch in enumerate(batches, start=1):
            try:
                result = _call_with_retry(clients, model, batch, prompt,
                                          stats=stats, gates=gates,
                                          progress=progress)
            except Exception as e:
                _fail_batch(batch, n_batches, num, e, progress)
                if progress is not None:
                    progress.adv("scoring", 1, f"lote {num} falhou")
                continue
            _apply_batch_result(batch, id_map, result, progress)
            if progress is not None:
                progress.adv("scoring", 1, f"lote {num}/{n_batches}")
            else:
                print(f"   -> lote {num}/{n_batches} concluído")
    else:
        if progress is None:
            print(f"   -> scoring paralelo: {workers} workers "
                  f"({len(clients)} chaves, lote i → chave i%{len(clients)})")
        else:
            progress.detail(f"scoring paralelo: {workers} workers")
        lock = threading.Lock()

        def run_batch(num_batch: tuple[int, list]):
            num, batch = num_batch
            try:
                result = _call_with_retry(
                    clients, model, batch, prompt, stats=stats,
                    key_hint=(num - 1) % len(clients),
                    gates=gates, lock=lock, progress=progress)
            except Exception as e:
                return (num, (None, e))
            return (num, (result, None))

        # client OpenAI (httpx) é thread-safe; resultados aplicados pelo
        # índice do lote na thread principal — ordem de conclusão irrelevante.
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="score") as pool:
            done = dict(pool.map(run_batch, enumerate(batches, start=1)))
        for num in range(1, n_batches + 1):
            result, err = done[num]
            if err is not None:
                _fail_batch(batches[num - 1], n_batches, num, err, progress)
            else:
                _apply_batch_result(batches[num - 1], id_map, result, progress)
            if progress is not None:
                progress.adv("scoring", 1, f"lote {num}/{n_batches}")
            else:
                print(f"   -> lote {num}/{n_batches} concluído")
    wall = time.time() - t_wall

    if cache_dir and fingerprint:
        save_scores(cache_dir, fingerprint, candidates, progress)

    failed = sum(1 for c in candidates if c.failed)
    n_req = stats["requests"]
    avg = stats["total_time"] / n_req if n_req else None
    min_lat = stats.get("min_lat")
    max_lat = stats.get("max_lat")
    speedup = (stats["total_time"] / wall) if wall > 0 else None
    if progress is None:
        print(f"   -> scoring: {len(candidates)-failed} pontuados, {failed} sem nota")
        print(f"   -> [perf] scoring: {n_req} requests em {wall:.1f}s wall "
              f"(sequencial ~{stats['total_time']:.1f}s, {speedup:.2f}x com "
              f"{workers} worker(s)), lat req "
              f"{(min_lat or 0):.1f}/{avg or 0:.1f}/{(max_lat or 0):.1f}s "
              f"(min/med/max), retries={stats['retries']}, 429s={stats['rate_limited']}")
    else:
        progress.done("scoring",
                      f"{len(candidates)-failed} pontuados, {failed} sem nota "
                      f"({wall:.0f}s, {workers} workers, {stats['retries']} retries, "
                      f"{stats['rate_limited']}×429)")
        progress.detail(
            f"[perf] scoring: {n_req} requests em {wall:.1f}s wall "
            f"(sequencial ~{stats['total_time']:.1f}s, {speedup:.2f}x, lat "
            f"{(min_lat or 0):.1f}/{avg or 0:.1f}/{(max_lat or 0):.1f}s)")
    if metrics is not None:
        try:
            metrics.set_nvidia(
                model=model, requests=n_req, successes=stats["successes"],
                failures=stats["failures"], retries=stats["retries"],
                total_time_sec=round(stats["total_time"], 3),
                avg_latency_sec=round(avg, 3) if avg else None,
                prompt_tokens=stats["prompt_tokens"] or None,
                completion_tokens=stats["completion_tokens"] or None,
                total_tokens=stats["total_tokens"] or None,
                cost=None,  # sem tabela de preços confiável => nunca estimar
                keys=len(clients),
                workers=workers,
                min_latency_sec=round(min_lat, 3) if min_lat else None,
                max_latency_sec=round(max_lat, 3) if max_lat else None,
                rate_limited=stats["rate_limited"],
            )
            metrics.add_retries(stats["retries"])
        except Exception:
            pass
    return candidates
