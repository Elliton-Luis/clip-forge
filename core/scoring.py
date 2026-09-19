"""scoring.py — prompt, chamada NIM e pontuação em lotes.

SRP: só scoring. OCP: prompt extensível via context/examples sem editar código.
"""
import json
import math
import os
import re
import sys
import time
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


def _next_client(clients: list):
    """Próximo cliente em round-robin. Retorna (client, idx); valor nunca logado."""
    global _key_cursor
    idx = _key_cursor % len(clients)
    _key_cursor += 1
    return clients[idx], idx


def _call_with_retry(clients, model: str, batch: list, prompt: str, retries: int = 3,
                     stats: dict | None = None) -> dict:
    # clients: lista não-vazia de OpenAI; cada tentativa (incluindo retries
    # após 503) usa a próxima chave — quota por conta diluída entre chamadas.
    backoffs = [10, 30, 60]
    last = None
    nkeys = len(clients)
    for attempt in range(retries):
        client, kidx = _next_client(clients)
        if stats is not None:
            stats["requests"] += 1
        t0 = time.time()
        try:
            result, tokens = _call_nim(client, model, batch, prompt)
            if stats is not None:
                stats["successes"] += 1
                stats["total_time"] += time.time() - t0
                if tokens:
                    for k_src, k_dst in (("prompt", "prompt_tokens"),
                                         ("completion", "completion_tokens"),
                                         ("total", "total_tokens")):
                        v = tokens.get(k_src)
                        if v is not None:
                            stats[k_dst] = (stats.get(k_dst) or 0) + v
            return result
        except json.JSONDecodeError as e:
            last = e
            print(f"      ! JSON inválido (tentativa {attempt+1}/{retries}, "
                  f"chave {kidx+1}/{nkeys}): {e}")
        except Exception as e:
            last = e
            print(f"      ! Erro NIM (tentativa {attempt+1}/{retries}, "
                  f"chave {kidx+1}/{nkeys}): {e}")
        if stats is not None:
            stats["total_time"] += time.time() - t0
            if attempt == retries - 1:
                stats["failures"] += 1
            else:
                stats["retries"] += 1
        if attempt < retries - 1:
            wait = backoffs[min(attempt, len(backoffs) - 1)]
            print(f"        aguardando {wait}s...")
            time.sleep(wait)
    raise last  # type: ignore


def score(candidates: list, model: str = DEFAULT_MODEL,
          cache_dir: Path | None = None, fingerprint: str | None = None,
          force: bool = False, context: str | None = None, examples_path: str | None = None,
          metrics=None) -> list:
    if cache_dir and fingerprint and not force:
        if load_scores(cache_dir, fingerprint, candidates):
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
    if len(clients) > 1:
        print(f"   -> rodízio entre {len(clients)} chaves NVIDIA (resiliência a quota/503)")
    examples = _load_examples(examples_path)
    prompt = _build_prompt(SYSTEM_PROMPT, context, examples)

    n_batches = math.ceil(len(candidates) / SEGMENTS_PER_SCORING_CALL)
    print(f"[3/5] Pontuando {len(candidates)} candidatos em {n_batches} chamada(s)...")
    id_map = {idx: c for idx, c in enumerate(candidates)}
    stats = {"requests": 0, "successes": 0, "failures": 0, "retries": 0,
             "total_time": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
             "total_tokens": 0}

    for start in range(0, len(candidates), SEGMENTS_PER_SCORING_CALL):
        batch = list(enumerate(candidates))[start:start + SEGMENTS_PER_SCORING_CALL]
        ids = {idx for idx, _ in batch}
        try:
            result = _call_with_retry(clients, model, batch, prompt, stats=stats)
        except Exception as e:
            print(f"   ! Falha lote {start//SEGMENTS_PER_SCORING_CALL+1}/{n_batches}: {e}. Marcando sem nota.")
            for _, c in batch:
                c.failed = True
            continue
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
                print(f"      ! candidato {idx} sem retorno — sem nota")
        print(f"   -> lote {start // SEGMENTS_PER_SCORING_CALL + 1}/{n_batches} concluído")

    if cache_dir and fingerprint:
        save_scores(cache_dir, fingerprint, candidates)

    failed = sum(1 for c in candidates if c.failed)
    print(f"   -> scoring: {len(candidates)-failed} pontuados, {failed} sem nota")
    if metrics is not None:
        try:
            n_req = stats["requests"]
            metrics.set_nvidia(
                model=model, requests=n_req, successes=stats["successes"],
                failures=stats["failures"], retries=stats["retries"],
                total_time_sec=round(stats["total_time"], 3),
                avg_latency_sec=round(stats["total_time"] / n_req, 3) if n_req else None,
                prompt_tokens=stats["prompt_tokens"] or None,
                completion_tokens=stats["completion_tokens"] or None,
                total_tokens=stats["total_tokens"] or None,
                cost=None,  # sem tabela de preços confiável => nunca estimar
                keys=len(clients),
            )
            metrics.add_retries(stats["retries"])
        except Exception:
            pass
    return candidates
