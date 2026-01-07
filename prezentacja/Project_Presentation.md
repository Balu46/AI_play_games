# Prezentacja Projektu: Optymalizacja Agentów RL w Środowiskach Gier

---

## 1. Definicja Problemu (Problem Definition)
**Cel:**
Stworzenie i optymalizacja agentów Uczenia ze Wzmocnieniem (Reinforcement Learning), zdolnych do efektywnego rozwiązywania klasycznych problemów sterowania i nawigacji.

**Konkretne zadania:**
*   Autonomiczne sterowanie lądownikiem księżycowym (`LunarLander-v3`).
*   Wyścigi samochodowe z analizą obrazu (`CarRacing-v3`).
*   Balansowanie wahadłem (`CartPole-v1`).

**Dlaczego to ważne:**
Projekt buduje framework do automatycznego dobierania hiperparametrów, co jest kluczowe dla wdrażania systemów decyzyjnych w robotyce i systemach autonomicznych.

**Oczekiwane rezultaty:**
*   Wytrenowane modele (pliki `.zip`).
*   Raporty skuteczności (średnia nagroda).
*   Zautomatyzowany pipeline optymalizacyjny.

---

## 2. Gromadzenie Danych (Data Collection)
**Źródła danych:**
W przeciwieństwie do klasycznego ML, w RL "dane" pochodzą z interakcji agenta ze środowiskiem symulacyjnym biblioteki **Gymnasium**.

**Dostępność i jakość:**
*   Wykorzystano certyfikowane środowiska (OpenAI Gym/Gymnasium), co gwarantuje powtarzalność i wysoką jakość fizyki symulacji.
*   Zbiór danych jest nieskończony – generowany na bieżąco (on-policy) lub pobierany z bufora (off-policy).

**Eksploracja:**
*   `CartPole`: Wektor stanu (pozycja, prędkość, kąt).
*   `CarRacing`: Obraz RGB 96x96 pikseli (Top-down view).

---

## 3. Przetwarzanie Danych (Data Preprocessing)
**Przygotowanie wejścia dla modelu:**

1.  **Normalizacja:**
    *   Automatyczne skalowanie wartości pikseli z [0, 255] do [0, 1] dla sieci konwolucyjnych (CNN) w `CarRacing`.
    *   Normalizacja nagród (Reward Scaling) w celu stabilizacji gradientów.

2.  **Dostosowanie Przestrzeni Akcji:**
    *   **Wrapper (`DiscreteActionsWrapper`):** Konwersja ciągłych sygnałów sterujących (kierownica, gaz, hamulec) na dyskretne zestawy akcji dla algorytmu DQN.
    
3.  **Wektoryzacja Środowisk:**
    *   Użycie `DummyVecEnv` do równoległego przetwarzania i zarządzania resetowaniem środowiska.

---

## 4. Wybór Modelu (Model Selection)
Dla każdego środowiska przetestowano różne architektury w ramach biblioteki **Stable Baselines3**:

*   **Modele Bazowe:**
    *   **DQN (Deep Q-Network):** Używany jako baseline dla środowisk z dyskretną przestrzenią akcji.
*   **Modele Zaawansowane:**
    *   **PPO (Proximal Policy Optimization):** Główny "koń roboczy", stabilny algorytm on-policy, świetny do ciągłych i dyskretnych przestrzeni.
    *   **A2C (Advantage Actor Critic):** Wariant synchroniczny, szybszy, ale czasem mniej stabilny niż PPO.

---

## 5. Trenowanie Modelu (Model Training)
**Strategia podziału:**
*   **Training Set:** Środowisko z ziarnem losowości `seed=0`.
*   **Validation Set:** Środowisko ewaluacyjne z ziarnem `seed=42` (niewidziane podczas treningu).

**Optymalizacja Hiperparametrów:**
*   Wykorzystano framework **Optuna** do automatycznego tuningu:
    *   `learning_rate`: Skala logarytmiczna.
    *   `batch_size`, `gamma`, `net_arch` (rozmiar sieci).
    *   Specyficzne dla PPO: `ent_coef` (eksploracja), `clip_range`.

**Proces:**
*   Trening na CPU/GPU z wykorzystaniem PyTorch.
*   Mechanizm **Early Stopping**: Zatrzymanie treningu, gdy średnia nagroda przestaje rosnąć, aby uniknąć overfittingu do specyfiki symulacji.

---

## 6. Ewaluacja Modelu (Model Evaluation)
**Metryki:**
*   **Mean Reward:** Średnia suma nagród z epizodu.
*   **Episodic Length:** Czas trwania epizodu (np. czy lądownik się nie rozbił zbyt szybko).

**Metodologia:**
*   Okresowa ewaluacja (`EvalCallback`) co X kroków na oddzielnym środowisku testowym.
*   Zapisywanie tylko najlepszego modelu (`best_model.zip`).

---

## 7. Analiza Błędów (Error Analysis)
**Identyfikacja problemów:**
*   **Reward Plateau:** Sytuacje, gdzie agent utknął w lokalnym minimum (np. samochód kręci się w kółko, by zbierać małe nagrody, zamiast jechać trasą).
*   **Katastrofalne zapominanie:** Nagły spadek skuteczności podczas długiego treningu.

**Wnioski:**
*   Dla `CarRacing`: DQN ma trudności z płynnym sterowaniem bez odpowiedniego dyskretyzowania akcji (rozwiązano przez Wrapper).
*   Dla `LunarLander`: PPO radzi sobie znacznie lepiej z precyzyjnym lądowaniem niż proste strategie.

---

## 8. Optymalizacja i Ulepszenia (Model Optimization)
**Zastosowane techniki:**
1.  **Automatyzacja (Optuna):** Zastąpienie ręcznego dobierania parametrów przeszukiwaniem bayesowskim (TPE Sampler).
2.  **Architektura Sieci:** Testowanie różnych rozmiarów warstw (Small, Medium, Large) w pliku konfiguracyjnym.
3.  **Funkcje Aktywacji:** Eksperymenty z `ReLU`, `Tanh`, `ELU`.

**Wyniki:**
Uzyskano stabilniejszą zbieżność i wyższe wyniki końcowe po dobraniu `gamma` i `learning_rate` przez Optunę.

---

## 9. Dokumentacja Projektu (Documentation)
**Struktura:**
*   **Codebase:** Przejrzysty podział na `src/stable_baseline`, `src/bare_bone`.
*   **Instrukcje:** Plik `README.md` zawiera instrukcje instalacji, treningu i wizualizacji.
*   **Raport:** Niniejsza prezentacja podsumowująca podejście.

**Kluczowe artefakty:**
*   Skrypty `run.sh`, `run_optimize.sh`.
*   Plik konfiguracyjny `config.json`.

---

## 10. Podsumowanie (Final Presentation Summary)
**Osiągnięcia:**
*   Zbudowano kompletny pipeline treningowy RL w Pythonie.
*   Zintegrowano nowoczesne narzędzia (SB3 + Optuna).
*   Zademonstrowano skuteczność na trzech zróżnicowanych środowiskach.

**Wnioski:**
Algorytmy on-policy (PPO) okazały się najbardziej uniwersalne. Automatyzacja doboru hiperparametrów znacząco skróciła czas potrzebny na osiągnięcie zadowalających wyników.

**Dalsze kroki:**
*   Implementacja algorytmu SAC (Soft Actor-Critic) dla `CarRacing`.
*   Rozbudowa modułu wizualizacji o wykresy porównawcze w czasie rzeczywistym.
