import streamlit as st
import spacy
from spacy import displacy
from spacy.language import Language
import re

# Custom NER component
@Language.component("custom_ner_component")
def custom_ner_component(doc):
    patterns = {
        "numbers": {
            "integer": r'\b\d+\b',
            "decimal": r'\b\d+\.\d+\b',
            "negative": r'\b-\d+(\.\d+)?\b',
            "thousands_comma": r'\b\d{1,3}(,\d{3})+\b',
            "thousands_space": r'\b\d{1,3}( \d{3})+\b',
            "ordinal": r'\b\d+(?:st|nd|rd|th)\b',
            "cardinal": r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b',
            "roman": r'\b[MCDLXVI]+\b',
            "percentage": r'\b\d+%|\b\d+\spercent\b',
            "range_dash": r'\b\d+\s?[-–]\s?\d+\b',
            "range_to": r'\b\d+\s?(?:to)\s?\d+\b',
            "scientific_notation": r'\b\d+(\.\d+)?[eE][+-]?\d+\b',
            "phone_number": r'\b(?:\+?\d{1,3})?[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b',
            "time_format": r'\b\d{1,2}:\d{2}(:\d{2})?\b',
            "fraction": r'\b\d+/\d+\b',
        },
        "citations": {
            "square_brackets": r'\[\d+(?:-\d+)?(?:,\s?\d+)*\]',
            "round_brackets": r'\(\d+(?:-\d+)?(?:,\s?\d+)*\)',
            "curly_brackets": r'\{\d+(?:-\d+)?(?:,\s?\d+)*\}',
            "superscript": r'(?<!\w)\d+(?!\w)',
            "name_year": r'\b[A-Z][a-z]+,?\s?\(?\d{4}\)?(?:,\s?\d{4})*\b',
            "inline_name_year": r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?,?\s?\(?\d{4}\)?(?:,\s?\d{4})*\b',
        },
        "other_floats": {
            "supplementary_materials": r'\b(?:Supplementary|Supplemental)\s+(?:Material|Figure|Table)\s+\d+\b',
            "appendix": r'\b(?:Appendix|Appendices)\s+[A-Z]\b',
            "audio_video": r'\b(?:Audio|Video)\s+(?:File|Clip|Track|Recording)\s+\d+\b',
            "external_materials": r'\b(?:Supplementary|Appendix|Audio|Video)\s+(?:Materials?|Files?|Clips?|Tracks?|Figures?|Tables?|Appendices?)\s+[A-Z]?\d*\b',
            "range_materials": r'\b(?:Supplementary|Appendix)\s+(?:Material|Figure|Table|Appendix|Clip|File)\s+\d+(?:[-–]\d+)?\b',
            "named_materials": r'\b(?:Supplementary|Appendix|Audio|Video)\s+(?:Materials?|Files?|Clips?|Tracks?|Figures?|Tables?|Appendices?)\s+[A-Z]\b',
        },
        "si_and_time_units": r'''
            \b\d+(?:[\.,]\d+)?\s*
            (?:
                [kKmM]?  # Optional prefixes for kilo, mega, milli, etc.
                (?:  
                    meter|metre|m|gram|g|kilogram|kg|second|s|ampere|A|kelvin|K|mole|mol|candela|cd|joule|J|watt|W|newton|N|pascal|Pa|hertz|Hz|coulomb|C|volt|V|ohm|Ω|siemens|S|farad|F|henry|H|lux|lx|becquerel|Bq|gray|Gy|sievert|Sv|liter|L|l|radian|rad|steradian|sr|decibel|dB
                )  
                |
                (?:hour|h|minute|min|second|s)  # Time units
            )\b
        '''

       
    }
    
    entities = []
    for label, pattern_group in patterns.items():
        if isinstance(pattern_group, dict):
            for sublabel, pattern in pattern_group.items():
                matches = re.finditer(pattern, doc.text)
                for match in matches:
                    start = match.start()
                    end = match.end()
                    entity_label = f"{label.upper()}_{sublabel.upper()}"
                    span = doc.char_span(start, end, label=entity_label)
                    if span is not None:
                        entities.append(span)
        else:
            matches = re.finditer(pattern_group, doc.text, re.VERBOSE | re.IGNORECASE)
            for match in matches:
                start = match.start()
                end = match.end()
                span = doc.char_span(start, end, label=label.upper())
                if span is not None:
                    entities.append(span)
    
    entities = sorted(entities, key=lambda span: (span.start, -(span.end - span.start)))
    filtered_entities = []
    seen_tokens = set()
    for ent in entities:
        if not any(token in seen_tokens for token in range(ent.start, ent.end)):
            filtered_entities.append(ent)
            seen_tokens.update(range(ent.start, ent.end))
    
    doc.ents = filtered_entities
    return doc

# Initialize the spaCy model
nlp = spacy.blank("en")
nlp.add_pipe("custom_ner_component", last=True)

# Streamlit app
st.title("Rule-Based NER Visualization")

text = st.text_area("Enter Text:", height=200)

if st.button("Process Text"):
    doc = nlp(text)
    html = displacy.render(doc, style="ent", jupyter=False)
    st.write(f"<div style='font-size: 1.2em;'>{html}</div>", unsafe_allow_html=True)
