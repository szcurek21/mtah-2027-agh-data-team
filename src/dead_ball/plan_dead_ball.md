# Plan pracy: Dead Ball i infrastruktura

## 1. Zasady wynikające z `spellF1`

Moduł `D` nie powinien maksymalizować frame-accuracy. Powinien minimalizować liczbę błędnych zdarzeń: fałszywego podziału posiadania i scalenia dwóch posiadań.

- **Fałszywe `D` jest droższe niż pominięte `D`** w większości sytuacji. Jedno spurious `D` rozcina posiadanie na dwa fragmenty i traci się jedno TP, a przybywają dwa FP. Wyjątek: gdy po przerwie piłkę ma ta sama drużyna (np. aut dla Barcy po wybiciu piłki przez Barcę). Wtedy brak `D` scala dwa posiadania w jedno. Gdy drużyna się zmienia, decoder i tak rozdzieli posiadania po zmianie `H`↔`A`.
- **Najważniejsze są dokładne granice**, nie sama obecność `D`. Przy progach IoU 0.3–0.7 błąd ±1 s jest bez znaczenia dla 30-sekundowego posiadania, ale zabija posiadanie 3–4-sekundowe. Błąd granicy restartu bezpośrednio przesuwa początek następnego posiadania.
- **Moduł powinien zwracać segmenty z pewnością, nie tylko ramkę po ramce.** Decoder (`adriankulig13`) może wtedy traktować `D` jako twarde ograniczenie albo miękki prior.

## 2. Architektura `src/dead_ball/`: model kandydat → weryfikacja → granice

Nie ma etykiet treningowych, a 5 minut publicznego GT jest tylko do sanity checku. Dlatego sensowne jest podejście regułowo-probabilistyczne z samokalibracją, a nie klasyczny model nadzorowany.

**Warstwa A: struktura meczu (twarde).** Ramki przed rozpoczęciem meczu (`< 11185`), w trakcie przerwy (`90652–110807`) i po meczu (`> 190197`) dostają maskę `non_play`. To jest darmowa i pewna część.

**Warstwa B: kandydaci z Smart Tagging (wysoki recall).** Z ~638 zdarzeń bierzesz `Throw-in`, `Corner Kick`, `Goal Kick`, `Free Kick`, `Kick Off` i `GOALS`. Każde tworzy okno poszukiwań `[start - delta_1, end + delta_2]`. Tagi są szumowe, więc to są kandydaci, a nie prawda. *(Uwaga zespołowa: wszystkie 26 autów `Throw-in` ma w tagach `Team: N/A` – dla modułu `D` to bez znaczenia, bo piłka jest martwa, ale informacja ta jest kluczowa dla `mrudnicki6` przy przejęciu posiadania po wznowieniu).*

**Warstwa C: weryfikacja i dokładne granice z trackingu.** Dla każdego okna szukasz dwóch punktów zmiany:

- **Początek `D`:** ostatnia ramka żywej gry. Sygnały: piłka przekracza linię (ostatnia widoczna pozycja blisko krawędzi boiska `[0, 1]` przed luką NaN), nagły spadek zbiorczej prędkości graczy, zatrzymanie piłki.
- **Koniec `D` (restart):** impuls przyspieszenia piłki, a gdy piłka jest NaN (41% ramek), skok prędkości kopiącego i zbiorcze przyspieszenie reszty (zawodnicy ruszają po kopnięciu).

Do detekcji punktów zmiany zacznij od prostej **histerezy** (próg wejścia ≠ próg wyjścia) lub wektorowego **CUSUM w czystym `numpy`/`scipy`** na wygładzonym sygnale `median_speed` 20 graczy z oknem ~1–2 s. To podejście jest błyskawiczne obliczeniowo (199k ramek), w 100% deterministyczne, łatwe do debugowania i nie wprowadza zbędnych zależności. Bardziej złożone metody segmentacji, np. PELT (`ruptures`), można potraktować jako późniejszy benchmark.

**Warstwa D: detektor kinematyczny jako uzupełnienie recall.** Łapie przerwy bez tagu (kontuzje, VAR, spalony). Działa z wysokim progiem pewności. Segment bez tagu trafia na wyjście tylko przy mocnym dowodzie: długa cisza kinematyczna plus brak piłki lub piłka stojąca.

**Warstwa E: fuzja.** Najprościej logistyczna fuzja dowodów w `p_dead(t)` plus segmentacja semi-Markovowska z minimalnymi czasami trwania:

- `D` krótsze niż ~2 s (50 ramek) odrzucone, chyba że jest mocno potwierdzone tagiem,
- luki `Live` w środku długiego `D` krótsze niż ~1 s zamykane.

## 3. Cechy, które warto policzyć

- `median_speed`, `p90_speed`, odsetek graczy poniżej 1.5 m/s w oknie 1–3 s.
- Rozrzut formacji i jej stacjonarność (ustawianie muru, zawodnicy ustawiają się w polu karnym).
- Ruch piłki: prędkość, przyspieszenie, odległość od linii, długość i pozycja luk NaN.
- **"Teleport piłki"**: różnica pozycji między ostatnią widoczną ramką przed NaN a pierwszą po. Duża wartość przy krótkiej luce sugeruje piłkę poza grą i wznowienie gdzie indziej.
- Odległość najbliższego gracza do piłki w momencie restartu (wykonawca).
- Cechy względne wobec tagu: czas od startu/końca tagu, typ zdarzenia.

## 4. Samokalibracja bez etykiet

To kluczowy trik, bo semantyka `start`/`end` w tagach jest nieznana i szumowa.

1. Dla każdego typu zdarzenia (np. ~100+ aut) wykryj kinematyczny punkt zmiany.
2. Policz rozkład `offset = punkt_zmiany - granica_tagu`.
3. Użyj mediany jako stałej korekty przesunięcia dla danego typu.

To nie używa GT, więc jest zgodne z regulaminem. Dodatkowo:

- **Nie dostrajaj progów na publicznych 5 minutach.** Zrób z nich wyłącznie test dymny formatu i sanity checku.
- **Ręczne etykietowanie oficjalnego meczu jest bezwzględnie zakazane** (`RULES.md`: *"No hand labels on the provided game may be used to train or validate a model"*). Nie wolno tworzyć ani uczyć modeli na ręcznych adnotacjach tego meczu. Natomiast **wzrokowa inspekcja** (oglądanie trajektorii, wykresów `median_speed` i wideo w celu debugowania algorytmu) jest w 100% legalna i wysoce wskazana.
- Narzędzie wizualne do inspekcji (wykres `median_speed`, tagów i `p_dead` na osi czasu plus podgląd wideo) bardzo przyspiesza debugowanie i jest w pełni legalne.

## 5. Anti-flicker i etykieta `X`

- Wymuś minimalną długość segmentów `D` i `Live` (patrz wyżej).
- **Pułapka etykiety `X` na granicy restartu:** W `EVALUATION.md` wyjaśniono, że oficjalny skorer przed obliczeniem `spellF1` wykonuje *forward-fill* na etykietach `X`. O ile `X` w głębi długiego posiadania lub w środku martwej piłki jest bezpieczny, o tyle postawienie `X` na granicy restartu (`D` -> `H`/`A`) spowoduje rozlanie (*bleed-over*) etykiety `D` w głąb akcji ofensywnej, opóźniając początek posiadania i psując IoU. Dlatego na przejściach stanów należy dążyć do twardej decyzji o klatce wznowienia, a nie `X`. To kluczowa kwestia do uzgodnienia z `adriankulig13` przy projektowaniu dekodera.

## 6. Infrastruktura

- **Determinizm:** `uv sync --frozen`, stały seed w jednym miejscu (`config`), `PYTHONHASHSEED=0`, jednowątkowość lub deterministyczne biblioteki, sortowanie przed każdą agregacją zależną od kolejności. CSV zapisuj ręcznie (`"\n"`, bez float, jawny nagłówek `frame,label`).
  - **Zaimplementowane:** `PYTHONHASHSEED` ustawiany w `main.py`, seed management, manifest z git head i lock hash
- **Walking skeleton od pierwszego dnia:** `main.py` uruchamia cały pipeline na atrapach (np. odtworzony `baseline_smooth50`) i przechodzi `validate_submission.py`. Każdy kolega podmienia swój moduł bez psucia reszty.
- **Cache etapów:** artefakty pośrednie w parquet/npz, klucz = hash konfiguracji + wersja kodu. Szybkie iteracje bez ponownego liczenia wszystkiego.
  - **Zaimplementowane:** `src/utils/cache.py` z funkcjami `cache_stage()`, `load_cached_stage()`, `clear_all_cache()`
  - Cache'uje wyniki dead ball w `.cache/` z hash konfiguracji jako klucz
  - CLI flags: `--clear-cache`, `--no-cache`
- **Testy:** `pytest` dla kontraktów (kształty, dtypes, zakresy), test integracyjny `main.py` → `validate_submission`, test "golden hash" (SHA-256 `submission.csv` dla dwóch kolejnych uruchomień musi być identyczny).
  - **Zaimplementowane:** `test_golden_hash_full_submission` w `tests/test_dead_ball.py`
  - Wszystkie testy przechodzą (23 testów, including calibration tests)
- **Jakość:** `ruff`, `mypy`/`pyright`, `pre-commit`, GitHub Actions z `uv sync --frozen` i testami.
  - **Zaimplementowane:** `ruff` z pre-commit hooks w `.pre-commit-config.yaml`
  - CI w `.github/workflows/ci.yml` - sprawdza linting, formatting, testy, golden hash, cache
- **Manifest uruchomienia:** commit, hash `uv.lock`, hash configu, seed zapisywane obok CSV.
  - **Zaimplementowane:** `src/utils/manifest.py` z `write_run_manifest()`
  - Manifest zapisywany jako `<submission>.manifest.json`

## 7. Kontrakt wyjściowy (szkic)

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np
import numpy.typing as npt

N_FRAMES = 199_319


@dataclass(frozen=True)
class DeadSegment:
    start: int  # pierwsza ramka D (włącznie)
    end: int  # ostatnia ramka D (włącznie)
    kind: Literal[
        "throw_in", "corner", "goal_kick", "free_kick", "kick_off", "goal", "other", "non_play"
    ]
    confidence: float  # [0, 1]
    start_sigma: float  # niepewność granicy w ramkach
    end_sigma: float


@dataclass(frozen=True)
class DeadBallOutput:
    p_dead: npt.NDArray[np.float32]  # (N_FRAMES,), P(piłka martwa)
    non_play: npt.NDArray[np.bool_]  # (N_FRAMES,), twarda maska struktury meczu
    segments: tuple[DeadSegment, ...]  # posortowane po start, bez nakładania
    restart_frames: npt.NDArray[np.int32]  # ramki restartu (koniec D)
```

Do modułu przekazujesz `TrackingData` i `Tags` (od `olgaerni`) oraz `FeatureBundle` (od `szcurek21`), a odbiera go decoder.

## 8. Kolejność prac

1. Szkielet repo, CLI, `uv`, walidacja, atrapy modułów (dzień 1–2).
2. Maska struktury meczu plus `D` wprost z tagów (surowe okna). To już daje bazowy wynik.
3. Kinematyczne wyrównanie granic i samokalibracja offsetów per typ zdarzenia.
4. Detektor uzupełniający i fuzja z filtrami anti-flicker.
5. Testy determinizmu, CI, golden hash.
6. Wspólna z decoderem ocena, czy `X` i miękkie `p_dead` poprawiają `spellF1`.

## Ryzyka do uzgodnienia z zespołem

- **Dryf zegara** między wideo, trackingiem i tagami (`olgaerni`): każdy stały offset niweczy wyrównanie granic. Warto go zmierzyć na początku.
  - **Zaimplementowane:** `compare_tag_clocks()` mierzy dryf, auto-korekcja dostępna
- **Format wymiany:** czy decoder chce segmentów, `p_dead`, czy obu (proponuję obu).
- **Lokalna metryka:** bez etykiet nie masz pełnego `spellF1`. Zbuduj lokalną implementację metryki ze `evaluation/` i używaj jej tylko na publicznych 5 minutach jako smoke test.

## Dodatkowe ulepszenia zaimplementowane (poza planem)

### 1. Per-event-type CUSUM thresholds
- **Problem:** Jednolite progi (threshold=4.0, drift=0.2) dla wszystkich typów zdarzeń
- **Rozwiązanie:** `DeadBallDetector.CUSUM_THRESHOLDS` z różnymi wartościami dla throw_in, corner, goal_kick, etc.
- **Benefit:** Lepsza detekcja granic dla różnych typów zdarzeń (throw-ins są szybsze niż goal kicks)

### 2. Auto-korekcja dryfu zegara
- **Problem:** Dryf między ElementTree a kloppy powoduje błędne granice
- **Rozwiązanie:** Jeśli dryf > threshold, auto-korekta używając mediany dryfu
- **Parametry:** `clock_drift_threshold=2`, `auto_correct_clock=True`
- **Benefit:** Granice są spójne z systemem Olgi

### 3. Samokalibracja offsetów tagów
- **Problem:** Tagi mogą być systematycznie przesunięte o kilka klatek
- **Rozwiązanie:** `calibrate_tag_offsets()` oblicza medianę offsetu dla każdego typu zdarzenia
- **Parametr:** `calibrate_tag_offsets=False` (włączony ręcznie, gdy masz tracking)
- **Benefit:** Automatyczna korekta systematycznych błędów tagów bez GT

### 4. Rozszerzone cechy kinematyczne
- **Problem:** Ograniczone cechy dla detekcji granic
- **Rozwiązanie:** Dodano `formation_spread` i `player_acceleration` do `TrackingSignals`
- **Benefit:** Lepsza detekcja ustawienia muru, impulsu restartu

### 5. Ulepszone ostrzeżenia
- **Problem:** Cichy fallback gdy brak XML - submission może wyglądać poprawnie ale nie mieć dead balls
- **Rozwiązanie:** Wyraźne WARNING z dużymi literami o braku Layer B
- **Benefit:** Użytkownik jest świadomy ograniczeń submission
