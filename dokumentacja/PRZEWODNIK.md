# 🎮 AI Play Games - Przewodnik Użytkownika

Witaj w projekcie **AI Play Games**! Ten projekt to zestaw inteligentnych agentów (AI), którzy uczą się grać w proste gry komputerowe metodą prób i błędów (Reinforcement Learning).

Nie musisz być programistą, aby uruchomić ten projekt. Ten przewodnik przeprowadzi Cię krok po kroku.

---

## 🚀 Szybki Start

### 1. Przygotowanie (Instalacja)
Zanim zaczniesz, musisz przygotować środowisko. Wystarczy uruchomić jeden skrypt, który zainstaluje wszystkie potrzebne biblioteki.

Otwórz terminal w folderze projektu i wpisz:
```bash
./setup.sh
```
*To może chwilę potrwać (zależnie od szybkości internetu).*

### 2. Uruchomienie Treningu
Aby AI zaczęło się uczyć grać, wpisz:
```bash
./run.sh
```
Co się stanie?
1. Projekt pobierze ustawienia z pliku `config.json`.
2. AI zacznie grać w wybraną grę (np. wyścigi samochodowe).
3. Na ekranie zobaczysz postępy (liczbę punktów, czas gry).
4. Po zakończeniu, system automatycznie wygeneruje wykresy wyników.

---

## ⚙️ Konfiguracja (Jak zmienić grę?)

Wszystkie ustawienia znajdują się w pliku `config.json`. Możesz go otworzyć w dowolnym edytorze tekstu.

Tak wygląda przykładowy plik:
```json
{
    "game": "car_racing",       <-- Tutaj wpisz nazwę gry
    "algorithms": ["ppo"],      <-- Algorytm uczący (polecamy "ppo")
    "total_timesteps": 1000000  <-- Jak długo AI ma się uczyć
}
```

### Dostępne gry:
- **`car_racing`**: Wyścigi samochodowe z widokiem z góry. Najtrudniejsza i najbardziej efektowna.
- **`lunar_lander`**: Lądowanie statkiem na księżycu. Średni poziom trudności.
- **`cart_pole`**: Balansowanie tyczką na wózku. Bardzo prosta i szybka do nauki (idealna na testy).

---

## 📊 Gdzie są moje wyniki?

Po zakończeniu treningu, zajrzyj do folderów wewnątrz katalogu gry (np. `car_racing/`):

1. **Wykresy wydajności**:
   - Folder: `debug_out/`
   - Znajdziesz tam pliki `.png` pokazujące jak AI stawało się coraz lepsze w czasie.

2. **Zapisane "Mózgi" AI**:
   - Folder: `models/`
   - Plik `best_model.zip` to najlepsza wersja AI, jaką udało się wytrenować.

---

## 🧠 Słowniczek (Dla ciekawskich)

- **Reinforcement Learning (RL)**: Metoda uczenia, w której AI dostaje "nagrodę" za dobre zachowanie (np. jazda po torze) i "karę" za złe (np. wypadnięcie z trasy).
- **Algorytmy**:
    - **PPO (Proximal Policy Optimization)**: Najbardziej stabilny i polecany algorytm. Uczy się ostrożnie, ale skutecznie.
    - **DQN (Deep Q-Network)**: Starszy, klasyczny algorytm. Czasem uczy się szybciej, ale bywa niestabilny.
    - **A2C (Advantage Actor Critic)**: Lżejsza i szybsza wersja, dobra do prostszych zadań.
- **Timesteps**: Liczba "klatek" lub decyzji, które podjęło AI. Im więcej, tym dłużej trwa nauka.

---

## 🛠 Rozwiązywanie problemów

- **"Permission denied" przy uruchamianiu skryptów**:
  Wpisz: `chmod +x setup.sh run.sh`

- **Trening trwa zbyt długo**:
  Zmniejsz liczbę `total_timesteps` w pliku `config.json` lub naciśnij `Ctrl+C` w terminalu (najlepszy model i tak zostanie zapisany, jeśli coś już się nauczył).
