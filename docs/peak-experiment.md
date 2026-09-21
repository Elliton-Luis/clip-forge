# Experimento: classic vs peak (2026-09-21)

Pergunta: "o novo algoritmo realmente encontrou momentos melhores?" —
respondida com números, não com "executou sem erro".
Artefatos em `debug/peak-lab/<video>/{classic,peak,compare}.json + compare.txt`.
Comando: `python clipper.py peak-compare VIDEO --top N --cache-dir .cache/clipper`.

Modo classic = pipeline atual (janelas até 90 s, rank por score médio).
Modo peak = experimental (`--selection-mode peak`): LLM julga subjanelas de
~12 s, o clip é construído ao redor do auge (frases inteiras + mínimo,
sem estofo até o máximo) e o rank é por intensidade do peak.

## VideoMedio.mp4 (7,2 min, 14 candidatos, 11 elegíveis, top 5 → 2 por diversidade)

| # | Classic (janela × score) | Peak (auge → contexto) |
|---|---|---|
| 1 | [89–179] **90 s** score 9.0 "O único que você conseguiu é pela pedra, pô" | janela 59–149 → peak [131–143] → clip **[119–144] 25 s** pscore 9.0/llm |
| 2 | [269–359] **90 s** score 8.0 "tem mais negros que morrem, né?" | janela 119–209 → peak [167–180] → clip **[167–198] 32 s** pscore 9.0/llm |

Durações: classic [90, 90] vs peak [24.7, 31.6]. O peak 1 está DENTRO da
janela do classic 1 (59–149 ∩ 89–179): mesmo momento, 90 s → 25 s ao redor
do auge ("É agora, agora é a hora"). Títulos ficaram da janela aqui porque
o re-título falhou com 503 da API (infra, não desenho) — melhoria de título
não comprovada nesta rodada.

## VideoLongo1.mp4 (26 min, 51 candidatos, top 6)

| Classic (90 s cheios) | Peak (clip / peak / pscore / título) |
|---|---|
| [0–90] 8.5 "Muteado essa merda fora" | [245–269] 24 s, peak [251–264] 9.5 "Tem que subir, será que dá o pulo?" [peak] |
| [329–419] 9.0 "Vão lá pra buchina!" | [275–296] 21 s, peak [281–294] 10.0 "Vamos lá, Meu Deus do céu, toma toma" [peak] |
| [719–809] 8.5 "Vai Lucas, vai Lucas!..." | [742–768] 26 s, peak [755–767] 9.0 "Pra cima, não quer descer, perdi a mochila" [peak] |
| [959–1049] 7.5 "Só pular aqui, ok?" | [779–804] 25 s, peak [791–803] 10.0 "Vai vai, pra cima, é verdade" [peak] |
| [1259–1349] 7.5 "Se tu matar Lucas bicho..." | [1313–1335] 22 s, peak [1319–1332] 8.5 "Se matar Lucas, juro por Deus" [peak] |
| [1529–1586] 57 s 7.5 "...chapéu de aviador..." | [1379–1400] 21 s, peak [1385–1398] 9.0 "Eu só morri, eu morri" [peak] |

Durações: classic [90×5, 56.8] vs peak [24.2, 20.9, 26.0, 24.8, 22.4, 21.1].
Detecção: 50 llm + 1 heurística; títulos: 6 do auge + 0 da janela (todos com
grounding validado contra o transcript). Peaks em 251, 281, 755, 791, 1319,
1385 — momentos distintos, sem duplicatas de mesmo auge. O peak também
ranqueou janelas de score médio 6.5 (filtro passa, intensidade decide) —
comportamento pretendido: auge forte em janela "ok" vence janela "boa" sem auge.

## Diversidade

Supressão por overlap de peak (≥ 50% = mesmo momento) provada em teste
(`test_suprime_mesmo_auge`: 3 janelas 10–70/20–80/30–90 do mesmo peak → 1
sobrevivente). No campo: nenhum par de peaks sobreposto foi selecionado.

## Limites honestos

- Custo: 1 chamada LLM por candidato (julgamento) + 1 por lote de títulos
  selecionados. Em VideoLongo1: 51 julgamentos. É modo experimental opt-in
  por esse motivo.
- Títulos do auge dependem da API no momento do run (503 → mantém janela
  com `title_source="window"` explícito, nunca inventa).
- Teto `--max-per-10min` continua valendo (VideoMedio: 1 bucket → 2 clips).
- End-to-end validado: `DerrubandoKit --selection-mode peak` renderizou
  [10.3–32] 22 s de janela [0–32] com peak [22.8–32], título do auge e bloco
  peak completo no manifest; manifests classic continuam byte-estáveis.
