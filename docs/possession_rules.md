# Zasady i Reguly Logiki Posiadania Pilki

Dokument podsumowuje reguly decyzyjne dla wyznaczania posiadania pilki w projekcie **More Than A Hack 2027 (Phase 1)**. Pelna specyfikacja modulu znajduje sie w [src/possession/README.md](../src/possession/README.md).

---

## 1. Cel i specyfika wyzwania
- **Predykcja:** Etykiety `H`, `A`, `D`, `X` dla kazdej z 199 319 klatek meczu.
- **Glowna metryka:** `spellF1` – srednia F1 z temporalnego IoU ciagow posiadania (progi 0.3, 0.4, 0.5, 0.6, 0.7).
- **Zlota zasada metryki:** Falszywy przeskok posiadania (np. na 0.3 s podczas walki stykowej) rozbija posiadanie i dramatycznie obniza `spellF1`. Posiadanie musi byc odporne na szum klatkowy.

---

## 2. Podsumowanie regul dla sytuacji stykowych (50-50, rykoszety, walka)

| Sytuacja boiskowa | Zachowanie fizyczne / cechy | Decyzja logiki posiadania |
|---|---|---|
| **Pojedynek 1 na 1 / Wslizg** | Obronica i napastnik blisko pilki ($< 1.5\text{ m}$) | Posiadanie **zostaje przy dotychczasowym posiadaczu**, dopoki odbierajacy nie ustabilizuje kontroli na $\ge 10$ klatek ($0.4\text{ s}$). |
| **Rykoszet / Musniecie** | Pilka uderza w obronce na krotko ($< 5$ klatek) i leci dalej z duza predkoscia ($> 3.5\text{ m/s}$) | **Brak zmiany posiadania**. Zdarzenie traktowane jako rykoszet niekontrolowany. |
| **Pojedynek powietrzny** | Pilka na wysokosci $z > 1.8\text{ m}$ lub znikajaca w gorze | Posiadanie przysluguje druzynie zagrywajacej, dopoki pilka nie zostanie opanowana na ziemi (*second ball*). |
| **Podanie w locie** | Pilka przelatuje blisko rywala, ale leci do partnera ($|\vec{v}| > 3.0\text{ m/s}$) | **Brak zmiany posiadania**. Obronica nie ma kontroli kinetycznej. |
| **Zamieszanie w polu karnym** | Pilka luzna, zaden gracz nie ma stabilnej kontroli | **Zasada inercji:** posiadanie zachowuje druzyna atakujaca, dopoki pilka nie zostanie wyekspediowana/opanowana. |
| **Brak detekcji pilki (41% klatek NaN)** | `ball_x` jest `NaN` w trakcie trwania gry | **Brak przejscia w stan `D`**. Podtrzymanie ostatniego znanego stanu (`H` lub `A`). |
| **Wznowienie ze stanu martwego (`D`)** | Pilka wraca do gry po przerwie | Pierwsze posiadanie otrzymuje **druzyna wznawiajaca gre** (ze Smart Taggingu lub pozycji przy linii/pilce). |

---

## 3. Inspiracja standardami rynkowymi
- **StatsBomb Two-Action Rule:** Wymog opanowania pilki i wykonania co najmniej dwoch kolejnych dzialan przez rywala, aby zaliczyc przejecie.
- **Opta Possession Chains:** Posiadanie traktowane jako ciagly lancuch zdarzen, odporny na pojedyncze musniecia i bloki obroncow.
- **Spearman Pitch Control:** Ocena mozliwosci dojscia do pilki na podstawie predkosci i kierunku ruchu gracza, a nie samej statycznej odleglosci.
