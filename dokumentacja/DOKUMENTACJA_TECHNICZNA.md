# 👨‍💻 Dokumentacja Techniczna (Wersja Rozszerzona)

Ten dokument stanowi szczegółowy przewodnik po kodzie źródłowym projektu **AI Play Games**. Jest przeznaczony dla programistów chcących zrozumieć "co dzieje się pod maską".

---

## 🏗 Architektura Systemu

Projekt oparty jest na bibliotece **Stable Baselines3** (SB3), która dostarcza gotowe implementacje algorytmów Reinforcement Learning (RL). Używamy **Gymnasium** jako standardu środowisk gier.

Całość spięta jest przez centralny punkt wejścia (`main.py`) i konfigurację w JSON.

### Przepływ danych (Data Flow)
1.  **Start**: `run.sh` uruchamia `main.py` z odpowiednimi flagami.
2.  **Konfiguracja**: `main.py` czyta `config.json` aby ustalić parametry (gra, algorytm, czas uczenia).
3.  **Trening**: `train.py` tworzy środowisko gry i agenta (Model), a następnie startuje pętlę uczenia.
4.  **Logowanie**: Podczas treningu, SB3 zapisuje metryki (TensorBoard) do folderu `logs/`.
5.  **Ewaluacja & Zapis**: Callbacki co pewien czas testują model. Najlepszy wynik jest zapisywany w `models/`.
6.  **Analiza**: Po treningu `plot.py` czyta logi i generuje wykresy `.png`.

---

## 📂 Szczegółowa Analiza Plików

### 1. `src/stable_baseline/main.py` (Entry Point)

Jest to główny sterownik aplikacji (Controller). Nie zawiera logiki biznesowej, a jedynie logikę sterującą.

**Kluczowe fragmenty:**
-   **`argparse`**: Definiuje dostępne polecenia CLI (`--mode`, `--env`, `--algo`). Pozwala nadpisywać ustawienia z pliku config z poziomu terminala.
-   **Ładowanie Configu**: Funkcja `load_config` wczytuje plik JSON. Jeśli podamy argumenty CLI, mają one pierwszeństwo przed JSON-em.
-   **Routing**: Blok `if args.mode == ...` decyduje, którą funkcję uruchomić:
    -   `train`: Uruchamia `train.py`.
    -   `visualize`: Uruchamia `visualize.py`.
    -   `plot`: Uruchamia `plot.py`.
    -   `optimize`: Uruchamia `optimize.py` (Optuna).

### 2. `src/stable_baseline/utils/train.py` (Trening)

To najważniejszy plik w projekcie. Odpowiada za konfigurację i uruchomienie procesu uczenia.

**Główne funkcje:**
-   **`train(...)`**: Ta funkcja spina wszystko w całość.
    1.  **Tworzenie środowiska**: `make_vec_env` tworzy instancję gry. Używamy `DummyVecEnv`, co jest standardem w SB3 (nawet dla jednej gry środowisko jest "wektoryzowane", czyli opakowane w listę).
    2.  **Inicjalizacja Agenta**: Na podstawie `ALGO_MAP` (słownik mapujący nazwy na klasy np. `PPO`, `DQN`) tworzony jest obiekt algorytmu.
    3.  **Hiperparametry**: Jeśli podano `hyperparams` (np. z optymalizacji), domyślne ustawienia (learning rate, gamma) są nadpisywane.
    4.  **Callbacki**:
        -   `EvalCallback`: Odpala osobne środowisko (`eval_env`) co X kroków, by sprawdzić jak model sobie radzi bez szumu eksploracji. To on zapisuje `best_model.zip`.

**Ważne koncepty**:
-   **Policy**: `MlpPolicy` to sieć neuronowa operująca na liczbach (np. pozycja wózka). `CnnPolicy` to sieć konwolucyjna (do obrazów), używana np. w `CarRacing`.
-   **Wrapper**: Dla DQN w CarRacing używamy `DiscreteActionsWrapper`, bo DQN nie obsługuje ciągłego sterowania (kierownica -1.0 do 1.0), więc musimy je zamienić na dyskretne (lewo, prawo, gaz).

### 3. `src/stable_baseline/visualization/plot.py` (Wykresy)

Odpowiada za wizualizację postępów. Nie korzysta z `tensorboard` w przeglądarce, ale generuje statyczne obrazki `.png` do raportów.

**Jak to działa:**
1.  Skrypt szuka plików `events.out.tfevents...` w katalogu `logs/`. Są to binarne pliki zapisu protokołu Protocol Buffers używane przez TensorBoard.
2.  `EventAccumulator`: Klasa z biblioteki TensorFlow/TensorBoard, która parsuje te pliki.
3.  **Ekstrakcja danych**: Wyciągamy tagi takie jak `rollout/ep_rew_mean` (średnia nagroda w epizodzie) lub `train/loss`.
4.  **Seaborn/Matplotlib**: Dane trafiają do biblioteki graficznej, która rysuje wykresy liniowe porównujące różne algorytmy.

### 4. `src/stable_baseline/optimize.py` (Optymalizacja)

Ten moduł używa biblioteki **Optuna** do automatycznego szukania najlepszych parametrów (Hyperparameter Tuning).

**Logika:**
-   **`objective(trial)`**: To funkcja celu. Optuna "wymyśla" zestaw parametrów (np. `learning_rate=0.001`), uruchamia trening (`train()`) i zwraca uzyskany wynik (nagrodę).
-   **Przetrzesz**: Optuna na podstawie historii prób zgaduje, jakie parametry mogą dać lepszy wynik w kolejnej próbie (używa estymatora TPE - Tree-structured Parzen Estimator).
-   Wyniki są zapisywane w bazie SQLite (`optuna.db`) oraz jako plik JSON (`best_params.json`), który potem może być użyty przez `main.py` i `train.py`.

### 5. `src/stable_baseline/visualization/visualize.py` (Podgląd)

Służy do "oglądania" jak gra nauczony model.

**Kluczowa pętla:**
```python
while not done:
    action, _ = model.predict(obs, deterministic=True) # Zapytaj model o akcję
    obs, reward, terminated, truncated, info = env.step(action) # Wykonaj akcję w grze
```
Zwróć uwagę na `deterministic=True`. Podczas treningu model czasem losuje akcje (eksploracja), ale podczas testów/pokazu chcemy, by grał najlepiej jak umie, więc wyłączamy losowość.

---

## 🛠 Rozszerzanie Projektu (Poradnik)

### Jak dodać nowy algorytm (np. SAC)?
1.  Zaimportuj go w `utils/train.py`: `from stable_baselines3 import SAC`.
2.  Dodaj do słownika `ALGO_MAP` w `train.py` i `visualize.py`.
3.  Pamiętaj, że SAC działa tylko z ciągłą przestrzenią akcji (jak CarRacing), a nie dyskretną (jak CartPole), chyba że użyjesz specjalnego wrappera.

### Jak zmienić architekturę sieci neuronowej?
W `train.py` modyfikujemy zmienną `net_arch`.
-   `[64, 64]` oznacza dwie warstwy ukryte po 64 neurony.
-   Możesz to zmienić w `setup` modelu, w argumencie `policy_kwargs`.

### Debugowanie problemów z treningiem
Jeśli model się nie uczy (wykres jest płaski):
1.  Sprawdź `learning_rate` (może być za duży lub za mały).
2.  Sprawdź `ent_coef` (parametr entropii - jeśli jest za mały, model za szybko "zdecyduje" że znalazł rozwiązanie i przestanie próbować nowych rzeczy).
3.  Zwiększ `n_steps` lub `batch_size`.
