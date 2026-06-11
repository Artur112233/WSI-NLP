# Analiza nastawienia autorów artykułów - raport z projektu NLP

#### Raport z realizacji projektu nr 3 z przedmiotu WSI - NLP analiza sentymentu

#### Autorzy - Artur Szabelski, Jerzy Wąsiewicz

# Wstęp

## Opis problemu

Podjęty przez nas problem należy do klasycznych zagadnień dziedziny Natural Language Processing. Analiza sentymentu pozwala ocenić nastawienie autora wypowiedzi. Najczęściej wykorzystuje się ją do oceny recenzji oraz komentarzy - w prosty sposób czy autor wypowiedzi pochwala czy krytykuje. Podobnie można potraktować artykuły - niektóre stanowią wprost recenzję, publicystyka wprost komentuje rzeczywistość. Wiadomości również są nacechowane nastawieniem autora. Istnieją różne metody i narzędzia analizujące sentyment, od dawna jest to wykorzystywane w kontekście recenzji i komentarzy w internecie. Ocena sentymentu w artykułach też nie jest czymś nowatorskim.

## Cel projektu

W ramach projektu postanowiliśmy zastosować metody oceny sentymentu w odniesieniu do artykułów. Efektem działania jest przyporządkowanie artykułu do jednej z trzech kategorii nastawienia autora - pozytywne, negatywne, neutralne. Po wytrenowaniu modelu na dostępnych w internecie datasetach przygotowaliśmy prosty program, któremu można podać artykuł w postaci tekstu lub linku i zwróci on informację o ocenie nastawienia autora przez model. Przygotowaliśmy model dla języka angielskiego, ze względu na powszechność jego użycia i dostępność danych oraz narzędzi.

# Metody i dane

## Metody analizy i klasyfikacji

Do opracowywanego problemu można podejść na dwa sposoby - LLM lub wektoryzacja i klasyfikacja. Zdecydowaliśmy się wykorzystać to drugie podejście - dla prostoty zrozumienia tematu na początek przygody z zagadnieniami NLP. Proces stworzenia naszego modelu składa się z:

### Preprocessingu tekstu

Wykorzystaliśmy narzędzie spaCy dostępne w Pythonie w celu dokonania lematyzacji. Uzyskujemy w ten sposób tekst dobry do podjęcia jego wektoryzacji. W toku eksperymentów najlepsze efekty dała prosta lematyzacja - lematyzacji podlegają wszystkie słowa, bez np. usuwania stop-words. (mimo to w trakcie prac dostosowaliśmy listę stop-words poprzez wyłączenie z niej tych mówiących o nacechowaniu (np. no, don't))

### Wektoryzacji

Wykonujemy wektoryzację TF-IDF - gotową implementację z pakietu sklearn.

### Klasyfikacji

Mamy 3 klasy docelowe nastawienia artykułu - positive, neutral, negative. Dla tych klas oraz zwektoryzowanych tekstów trenujemy model klasyfikacyjny. W drodze eksperymentów w gotowym modelu postawnoiliśmy zostawić Logistic Regression z parametrami dostosowanymi przez Grid Search. Sprawdziliśmy też modele LinearSVC i Random Forest.

## Datasety 
Do projektu wykorzystaliśmy

## Końcowy program

W efekcie prac i treningów uzyskaliśmy model dobrze działający na zlematyzowanym tekście. Dokonuje on wektoryzacji TF-IDF oraz klasyfikacji metodą Logistic Regression. Został on zapisany i może być załadowany w programie. Do modelu dobudowaliśmy prostą aplikację konsolową, do której można wkleić treść artykułu do oceny lub podać link do artykułu w internecie (wydobywany za pomocą narzędzia trafilatura), artykuł zostaje przepuszczony przez preprocessor i model i aplikacja zwraca wynikowe określenie nastawienia artykułu.

# Wyniki i obserwacje

## Ocena wyników treningu i działania modelu

## Obserwacje

# Podsumowanie

## Wnioski

## Pole do rozwoju

# Odwołania

## Źródła danych

## Biblioteki
