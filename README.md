# AGH Data Team: More than a Hack 2027, Phase 1: Ball Possession

## Cel
Jedno zdanie: dla każdej klatki meczu przewidujemy H/A/D/X; ranking: spellF1.

## Zespół i role
olgaerni - Dane i EDA
szcurek21 - Cechy
adriankulig13 - Dekoder czasowy
mrudnicki6 - Logika posiadania
PSzarkowicz - Martwa piłka i infrastruktura

## Szybki start
uv sync
(jak ustawić dane: rozpakować do data/ lub ustawić MTAH_DATA_DIR)
UWAGA: dane meczu nie są w repo i nie wolno ich tu dodawać.

## Uruchomienie pipeline'u
(komenda generująca CSV, uzupełnicie, gdy będzie gotowa;
 ta komenda jest też wymagana w zgłoszeniu)

## Struktura repo
src/ (moduły), notebooks/, results/ (CSV niecommitowane), docs/

## Kontrakt między modułami
tabela: moduł, wejście (kolumny), wyjście (kolumny)

## Zasady pracy
gałęzie, Pull Requesty, brak commitów na main

## Zasady konkursu
bez ręcznego etykietowania; dane nie do zewnętrznych LLM;
repo musi odtwarzać plik co do bajta (stałe seedy, uv.lock)
