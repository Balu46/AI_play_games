# Instrukcja Kompilacji Prezentacji

Aby wykompilować plik `presentation.tex` do formatu PDF, użyj poniższego polecenia w terminalu (będąc w folderze `prezentacja`):

```bash
pdflatex -interaction=nonstopmode presentation.tex
```

Jeśli komenda nie działa, upewnij się, że masz zainstalowane wymagane pakiety:

```bash
sudo apt install texlive-latex-base texlive-latex-extra
```

Wygenerowany plik `presentation.pdf` pojawi się w tym samym katalogu.
