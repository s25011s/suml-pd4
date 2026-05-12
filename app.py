import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# 1. Konfiguracja strony
st.set_page_config(page_title="Mój Asystent AI", page_icon="🤖")

# 2. Tytuł i grafika
st.title("🤖 Laboratorium 5: Aplikacja NLP")
st.write("---")

# 3. Krótka instrukcja
with st.expander("ℹ️ Instrukcja obsługi"):
    st.write("""
        Ta aplikacja pozwala na:
        1. **Analizę emocji** - sprawdzisz czy tekst jest pozytywny czy negatywny (tylko jęz. angielski).
        2. **Tłumaczenie** - przetłumaczysz tekst z języka angielskiego na niemiecki.
        Wybierz opcję z menu po lewej lub poniżej, wpisz tekst i poczekaj na wynik.
    """)

# 4. Sidebar (menu boczne)
st.sidebar.header("Ustawienia")
option = st.sidebar.selectbox(
    "Co chcesz zrobić?",
    ["Wybierz opcję", "Analiza emocji (ENG)", "Tłumaczenie (ENG -> DE)"]
)

# 5. Logika aplikacji
if option == "Analiza emocji (ENG)":
    st.subheader("Analiza wydźwięku tekstu")
    text = st.text_area("Wpisz tekst po angielsku:")
    if st.button("Analizuj"):
        if text:
            with st.spinner('Trwa analiza...'):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)
                st.success("Gotowe!")
                st.write(answer)
        else:
            st.warning("Najpierw wpisz jakiś tekst!")

elif option == "Tłumaczenie (ENG -> DE)":
    st.subheader("Tłumacz Angielski -> Niemiecki")
    text_to_translate = st.text_area("Wpisz tekst po angielsku:")

    if st.button("Tłumacz"):
        if text_to_translate:
            with st.spinner('Trwa tłumaczenie...'):
                try:
                    # Załadowanie modelu i tokenizera (bezpośrednio)
                    model, tokenizer = load_translation_model()

                    # Przygotowanie tekstu (tokenizacja)
                    inputs = tokenizer(text_to_translate, return_tensors="pt")

                    # Generowanie tłumaczenia
                    outputs = model.generate(**inputs)

                    # Zamiana numerów na tekst
                    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    st.success("Gotowe!")
                    st.info(f"Wynik: {decoded}")
                except Exception as e:
                    st.error(f"Błąd: {e}")
        else:
            st.warning("Najpierw wpisz tekst!")

# 6. Stopka z numerem indeksu
st.write("---")
st.caption("Autor: [TWÓJ NUMER INDEKSU TUTAJ]")