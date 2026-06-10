import text_analyser as ta


def main():
    print("=" * 45)
    print("      SYSTEM ANALIZY SENTYMENTU NLP      ")
    print("=" * 45)
    run_program = True

    analyser = ta.SentimentAnalyzer()

    while run_program:
        print("\nWybierz tryb działania:")
        print("2 - Analiza wpisanego tekstu")
        print("1 - Analiza artykułu z linku (URL)")
        print("0 - Zakończ program")

        input_mode_choice = input("\nTwój wybór (2/1/0): ")
        match input_mode_choice:
            case "2":
                text = input("Wpisz tekst do analizy: ")
                text = text.replace('"', '').replace("'", "").strip()

                if not text:
                    print("Tekst nie może być pusty!")
                    continue

                sentiment = analyser.analyse_text(text)
                print("Według modelu ten artykuł jest: ", sentiment)

            case "1":
                url = input("Wklej link do artykułu: ")
                if not url.strip():
                    print("Link nie może być pusty!")
                    continue

                sentiment = analyser.analyse_article_url(url)
                print("Według modelu ten artykuł jest: ", sentiment)

            case "0":
                run_program = False
                print("Zamykanie programu.")
            case _:
                print("Nieprawidlowy wybór")

if __name__ == '__main__':
    main()