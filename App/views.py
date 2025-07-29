import base64
import html
import io
import json
import re
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import pandas as pd
import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from PIL import Image


SEC_MARKER_FMT = "<!--sec:{slug}-->"   # HTML comment marker used to mark section boundaries
WORD_TOLERANCE = 0.8                    # Accept 80% of target words before regen; tune as needed

# Track token usage (optional debug / analytics)
TOTAL_TOKENS = 0
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0

GPT4O_INPUT_COST_PER_M = 2.50  # USD per 1M tokens
GPT4O_OUTPUT_COST_PER_M = 10.00  # USD per 1M tokens

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def chat_complete(messages: List[dict], *, model: str = "gpt-4o", temperature: float = 0.8,
                  top_p: float = 1.0, frequency_penalty: float = 0.15,
                  presence_penalty: float = 0.25, max_tokens: Optional[int] = None) -> str:
    global TOTAL_TOKENS, TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
    )
    out = resp.choices[0].message.content or ""
    # print('************************************************')
    # print(out)
    # print('*************************************************')
    usage = getattr(resp, "usage", None)
    if usage:
        TOTAL_TOKENS += getattr(usage, "total_tokens", 0)
        TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
        TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
    return out.strip()


def normalize_slug(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

#german
def build_longform_prompt(page_title: str, keywords: str, sections_rules: List[Tuple[str, int, str]], language: str) -> str:
    section_lines = []
    prompt = ""
    if language == 'deutsch':
        for slug, word_count, rule in sections_rules:
            rule_text = rule if rule else "Keine speziellen Anforderungen"
            section_lines.append(f"- {slug} ({word_count} Wörter): {rule_text}")
        prompt = f"""
        
                # ======================================
                # 0. VARIABLEN & KAPITELVORGABEN
                # ======================================

                MAIN_KEYWORD  = {page_title}
                KEYWORDS      = {keywords}           
                SECTION_LINES :
                
                {section_lines}     

                • Gib für jede SECTION_LINE eine prägnante H2-Überschrift aus – leserorientiert, konkret, mit passenden Keywords.
                • Keine generischen Titel wie „Kontext“, „Relevanz“, „Ablauf“ oder SEO-Phrasen aus Substantiven.
                • Die Überschrift soll Interesse wecken, aber nicht den Zweck beschreiben.

                ---

                # ======================================
                # 1. ZIEL
                # ======================================

                • Verfasse einen hochwertigen HTML-Servicetext aus Unternehmenssicht zum Thema "{page_title}".
                • Der Text dient als professionelle, unternehmensnahe Darstellung unserer Dienstleistungen.
                • Ziel ist es, den konkreten Mehrwert unseres Angebots verständlich, glaubwürdig und kundenorientiert darzustellen – ohne PR-Sprache, aber mit klarer Positionierung.
                • HTML-Ausgabe gemäß Abschnitt 9.

                ---

                # ======================================
                # 2. PRIORITÄT (P1–P3)
                # ======================================

                P1 – HTML-Struktur & Keywordplatzierung  
                P2 – Logischer Aufbau & Absatzverknüpfung  
                P3 – Sprachstil & Leserführung  

                ---

                # ======================================
                # 3. WORKFLOW
                # ======================================

                PHASE 1 – COMPOSE:  
                Verwende folgende Platzhalterstruktur:  
                [H1], [H2], [P], [UL], [LI]  

                PHASE 2 – MARKUP (temp ≤ 0.2):  
                Ersetze Platzhalter durch finalen HTML-Code. **Nur HTML ausgeben.**

                ---

                # ======================================
                # 4. CHAIN-OF-THOUGHT (intern)
                # ======================================

                1) Themenverständnis  
                2) Strukturierung der Abschnitte  
                3) Textproduktion (compose)  
                4) Qualitätsscan  
                5) HTML-Umsetzung (markup)
                
                ---

                # ======================================
                # 5. KEYWORD-REGELN (P1)
                # ======================================

                • {page_title}: in <h1>, ≥ 3× im Text (≤100 Wörter Abstand)  
                • KEYWORDS: mind. 1× pro Abschnitt (in <h2> oder <h3>), organisch eingebaut  
                • Keine Abwandlungen: exakt, orthografisch korrekt, natürlich integriert  

                ---

                # ======================================
                # 6. STIL-REGELN (P2)
                # ======================================

                ## A. Haltung & Ton

                - Sie-Form – ruhig, professionell, sachlich-empathisch
                - Keine künstliche Lockerheit
                - Leseransprache auf Augenhöhe
                - – Ziel: Orientierung durch nachvollziehbaren, glaubwürdigen Mehrwert – ohne Übertreibung
                - Kurze, realistische, allgemein erkennbare Alltagsszenen erlaubt  
                → Keine Einzelfälle, aber illustrative Mini-Situationen zur Verdeutlichung  
                → Ergänze nachvollziehbares Insiderwissen
                - Verwenden Sie lebensnahe, bildhafte, aber sachlich bleibende Formulierungen – mit feiner Emotionalität, ohne Übertreibung oder anbiedernden Ton.


                ## B. Positionierung 
                • Der Text soll nicht nur informieren, sondern die Leistungen und Stärken des Unternehmens klar erkennbar machen.
                • Positionieren Sie das Unternehmen als kompetenten, verlässlichen Anbieter – faktenbasiert, glaubwürdig und kundennah.
                • Vermeiden Sie übermäßige Selbstbezüge wie „unser Team“, „unsere Erfahrung“, „unsere Kund:innen“ – verwenden Sie neutrale Formulierungen oder bauen Sie die Leistung in den Lösungskontext ein (z.B. „Der gesamte Ablauf wird koordiniert“ statt „Unser Team koordiniert …“).
                    - Maximal 1× „unser/unserer“ pro Absatz.
                • Zeigen Sie konkret, **welche Probleme gelöst werden**, **wie der Service funktioniert** und **welchen Nutzen Kund:innen daraus ziehen**.
                • Verwenden Sie sachlich überzeugende Formulierungen
                • Bleiben Sie dabei präzise und zurückhaltend – keine PR-Sprache, keine Superlative, keine Heilsversprechen.

                ## C. Ausdruck

                - Klar, strukturiert, präzise – ohne Schnörkel
                - Keine Floskeln, Buzzwords, Superlative oder PR-Sprache
                - Keine Phrasen wie „Keine Sorge“, „Hand aufs Herz“
                - Keine Wortspiele oder Metaphern
                - Nutzen konkret benennen – keine Konjunktive
                - Vermeiden Sie ausgeleierte Werbeformulierungen wie „Erleben Sie“, „Vertrauen Sie“, „Mit uns wird ...“, „steht Ihnen mit Rat und Tat zur Seite“.
                - Wenig bis keine Dopplungen in Formulierungen wie „stressfrei und sorgenfrei“, „schnell und unkompliziert“.
                - Bevorzugen Sie klare, belegbare Aussagen statt Versprechens-Rhetorik.


                ## D. Satzbau & Rhythmus

                - Ø-Satzlänge 12–25 Wörter (StdAbw ≥ 6 W.)
                - 40 % kurz (≤12 W.), 40 % mittel (13–20 W.), 20 % lang (>20 W.)
                - Max. 1 Hauptsatz + 1 Nebensatz pro Satz
                - Nach zwei langen Sätzen: 1 kurzer Satz (<10 W.)
                - Satzanfänge variieren, kein gleichförmiger Rhythmus
                - Erlaubt sind gelegentliche stilistische Abweichungen wie Inversionen, Parenthesen oder Ein-Wort-Sätze, um einen lebendigeren Sprachfluss zu erzeugen.
                - Erzielen Sie einen lebendigen Rhythmus durch den Wechsel von kurzen Statements und längeren Gedankenbögen („Burstiness“).



                ## E. Struktur

                - Logisch aufeinander aufbauende Kapitel
                - Keine separaten Einleitungen je Abschnitt
                - Rhetorische Fragen: max. 2–3 pro Text, gezielt platziert, um Denkimpulse oder sanfte Übergänge zu schaffen – nicht zur Dramatisierung.
                - Kein Pathos, kein Storytelling um des Effekts willen
                - Der erste Satz jedes Abschnitts muss sich sprachlich klar vom vorherigen unterscheiden.
                - Vermeiden Sie gleichförmige Satzanfänge wie „Ein …“, „Der …“, „Ein weiterer …“, „Dabei …“ etc.
                    • Erlaubt sind z.B.:
                    – Beobachtungen
                    – logische Übergänge 
                    – kurze Feststellungen 
                    – situative Sätze 
                    • Verwenden Sie unterschiedliche Einstiegstypen:
                    - sachliche Feststellungen  
                    - kurze Problem-Skizzen  
                    - logische Übergänge  
                    - Kontextbezüge  
                - Variieren Sie Einleitungen bewusst in Rhythmus, Struktur und Perspektive.
                - Ziel hierbei ist ein lebendiger, dynamischer und abwechslungsreicher Textfluss – kein repetitiver Rhythmus trotz fester Struktur.
                - Vermeiden Sie inhaltliche Wiederholungen: Jeder Abschnitt soll einen eigenen, klar abgegrenzten Aspekt behandeln.
                - Nutzenargumente, Vorteile oder Prozessschritte dürfen nicht doppelt aufgeführt werden – weder sprachlich identisch noch inhaltlich redundant.
                - Achten Sie auf inhaltliche Fortschritte zwischen den Kapiteln – keine Rückgriffe auf bereits behandelte Punkte, außer zur gezielten Verknüpfung.


                ## F. Stilgrenzen

                - Keine CTAs wie „Jetzt anrufen“ oder „Jetzt buchen“
                - Kein aufgesetzter Humor
                - Keine Stereotype („der überforderte Kunde“)
                - Keine überzogenen Versprechen oder Heilsbotschaften
                - Keine erfundenen Geschichten (→ siehe Abschnitt HUMANIZER)

                ---

                # ======================================
                # 7. HUMANIZER (P3 – optional)
                # ======================================

                ## Ziel

                - Sachlich, praxisnah, erfahrungsbasiert – ohne Dramatisierung

                ## Darstellungsform

                - Verallgemeinerte Abläufe / Szenen erlaubt  
                → Keine konkreten Namen, Orte, Einzelfälle

                - Keine erfundenen Erfahrungsberichte
                - Keine persönlichen Anekdoten
                - Keine fiktiven Elemente oder Storytelling

                ## Wirkung

                - Der Text soll realistisch, glaubwürdig und nutzerorientiert wirken  
                - Ziel: fachlich fundierter, alltagsnaher Text mit hoher Praxisrelevanz

                ---

                # ======================================
                # 8. ANTI-BIAS & FAKTENCHECK
                # ======================================

                - Keine Stereotype; sensible Themen sachlich-empathisch behandeln
                - Fachliche Aussagen:
                → Wenn möglich: ≥ 2 Quellen
                → Alternativ: Abschwächung mit „oft“, „in vielen Fällen“, „meistens“

                ---

                # ======================================
                # 9. AUSGABEFORMAT (P1)
                # ======================================

                • Zulässige Tags: `<h1>`, `<h2>`, `<p>`, `<ul>`, `<li>`, `<br>`, `<!--sec:slug-->`  
                • Slug-Format: `^[a-z0-9-]+$`
                • Jeder <h2>-Abschnitt beginnt mit einem inhaltlich klaren, eigenständigen Eröffnungssatz – kein Rückgriff auf bereits verwendete Formulierungen oder Aussagen bzw. Wiederholung aus anderen Kapiteln.


                ### Struktur:

                1. 1. `<h1>` mit {page_title} + aussagekräftigem Slug sollte innerhalb des ersten Slugs stehen, wenn es sich um ein Banner handelt.  
                2. Optional: `<!--sec:banner-->`  
                3. beim Geben von {SEC_MARKER_FMT} ist der Slug der Abschnittsname, der in den Abschnittszeilen oder den Regeln erwähnt wird (Slug-Namen werden normalerweise wie Banner, (linkes_Bild_rechter_Inhalt,      linker_Inhalt_rechtes_Bild, linkes_Bild_rechter_Inhalt2, linker_Inhalt_rechtes_Bild, FAQ, weiterlesen usw.)            
                4. Für jede Zeile in SECTION_LINES:  
                `<!--sec:slug(section_name)--><h2>…</h2>` mit 150–200 Wörtern Text  
                5. Optional: `<!--sec:sources--><h2>Quellen</h2><ul><li>…</li></ul>`

                **Absatzregel:**  
            
                1. VSchreiben Sie vor jedem Abschnitt genau einen HTML-Kommentar {SEC_MARKER_FMT.format(slug='section_name_or_the_slug(eg:banner,left_image_right_content,left_content_right_image)')} mit dem entsprechenden Slug aus der unten stehenden Liste. Fügen Sie direkt danach eine sichtbare <h2>-Überschrift ein (mit Ausnahme der H1 am Anfang).

                # 10. QUALITÄTSKRITERIEN (P1–P3)

                • Jeder Abschnitt behandelt ein eigenständiges Thema ohne inhaltliche Dopplungen  
                • Satzanfänge, Satzlängen und Perspektiven variieren  
                • Keine übermäßigen Wiederholungen von „unser/unserer“  
                • Einstiegssätze sind abwechslungsreich und thematisch klar  
                • Sprache ist sachlich, klar, glaubwürdig – ohne PR-Floskeln  
                • Nutzen ist konkret und nachvollziehbar formuliert  
                • Textfluss ist logisch – keine isolierten oder losgelösten Absätze 
                • Keine allgemeinen Floskeln oder Pauschalaussagen – alle Aussagen sind plausibel, nachvollziehbar und realitätsnah
                • Sprachfluss wirkt natürlich und nicht „zu perfekt“ – stilistische Mikroabweichungen sind erlaubt  
                • Der Text vermeidet Roboterlogik oder monotone Syntax – Fokus liegt auf Leserfreundlichkeit und sprachlicher Authentizität   
                
                # 11.
                    Verfassen Sie den gesamten Inhalt in deutscher Sprache.
                    
            """
    elif language == 'englisch':
        for slug, word_count, rule in sections_rules:
            rule_text = rule if rule else "No special requirements"
            section_lines.append(f"- {slug} ({word_count} Words): {rule_text}")
        prompt = f"""
    
            # ======================================
            # 0. VARIABLES & SECTION GUIDELINES
            # ======================================

            MAIN_KEYWORD  = {page_title} 
            KEYWORDS      = {keywords}
            SECTION_LINES :

            {section_lines}  

            • Write a clear, reader-focused H2 heading for each SECTION_LINE – specific, benefit-oriented, keyword-integrated.  
            • No generic titles like "Context" or "Relevance" or keyword-dump phrases.  
            • Headlines should spark interest without explaining purpose.

            ---

            # ======================================
            # 1. OBJECTIVE
            # ======================================

            • Create a high-quality **HTML service description** from a business perspective about **{page_title}**.  
            • The text serves as a **professional representation of the service offering**.  
            • The goal is to convey the **concrete value** of the service clearly and credibly – without promotional tone but with confident positioning.  
            • Output the text in clean HTML format (see Section 9).

            ---

            # ======================================
            # 2. PRIORITY (P1–P3)
            # ======================================

            P1 – HTML structure & keyword placement  
            P2 – Logical structure & paragraph transitions  
            P3 – Tone & reader guidance

            ---

            # ======================================
            # 3. WORKFLOW
            # ======================================

            PHASE 1 – COMPOSE:  
            Use the following placeholder structure:  
            [H1], [H2], [P], [UL], [LI]

            PHASE 2 – MARKUP (temp ≤ 0.2):  
            Replace placeholders with final HTML. **Output only HTML.**

            ---

            # ======================================
            # 4. CHAIN OF THOUGHT (internal)

            1) Understand the topic  
            2) Structure content  
            3) Write each section  
            4) Quality scan  
            5) Apply HTML markup

            ---

            # ======================================
            # 5. KEYWORD RULES (P1)

            • {page_title}: in <h1>, ≥ 3× throughout the text (≤100 words apart)  
            • All other KEYWORDS: at least 1× per section (in <h2> or <h3>), placed naturally  
            • No variations: use keywords exactly as provided, with correct spelling and organic integration

            ---

            # ======================================
            # 6. STYLE RULES (P2)

            ## A. Tone & Voice

            - Use **professional, calm, respectful "you"-address**  
            - No artificial cheerfulness  
            - Speak to the reader as a peer  
            - Goal: Clarity, credibility, and relevance – no exaggeration  
            - Everyday situations are allowed but generalized – no storytelling  
            - Use precise, real-world phrasing – neutral but slightly emotionally aware  
            - Avoid hype, wordplay, or metaphors  
            - Use brief, illustrative examples (not case studies)

            ## B. Positioning

            • Focus on what problems are solved, how the service works, and what benefits customers gain  
            • Position the company as **reliable and competent**, not self-centered  
            • Use "our" or "we" only **once per paragraph**, if at all  
            • Reframe team-centric actions into solution-centric outcomes (e.g., “The process is coordinated” rather than “Our team coordinates it”)  
            • Avoid sales phrases or promotional claims

            ## C. Wording

            - Clear, specific, and structured  
            - Avoid fluff, jargon, and generic phrases  
            - No buzzwords like “effortless,” “worry-free,” “fast and easy”  
            - Avoid empty CTA language like “Call us now” or “Experience our service”  
            - State benefits factually, not hypothetically  
            - Avoid repeated dualities (“smooth and seamless,” “easy and efficient”)  
            - Focus on demonstrable, grounded value

            ## D. Sentence Structure

            - Avg. sentence length: 12–25 words  
            - Target: 40% short (≤12), 40% medium (13–20), 20% long (>20)  
            - Max: 1 main + 1 subordinate clause per sentence  
            - After two long sentences, insert one short (<10 words)  
            - Vary sentence starters to avoid monotony  
            - Use occasional rhythm breaks for liveliness (e.g., inversion, one-word sentences)

            ## E. Section Structure

            - Each section builds on the previous one  
            - Avoid repeating content or phrases  
            - Each section should bring **new information or perspective**  
            - Transitions should feel **natural and logical**  
            - Use varied entry styles: facts, transitions, observations  
            - Use rhetorical questions sparingly (2–3 max across full text), only to spark thought  
            - Do not reintroduce previously covered points unless to build on them  

            ## F. Style Boundaries

            - No CTAs  
            - No artificial humor  
            - No stereotypes (e.g., “the overwhelmed customer”)  
            - No idealized promises or exaggerated language  
            - No personal stories or fictive anecdotes

            ---

            # ======================================
            # 7. HUMANIZER (P3 – optional)

            ## Objective

            - Stay factual, practical, relatable – avoid dramatization

            ## Allowed

            - Generalized scenarios or routines  
            - No names, places, or fictionalized reviews  
            - Text should **feel realistic and grounded**

            ---

            # ======================================
            # 8. ANTI-BIAS & FACT CHECK

            - Avoid stereotypes; use sensitive, neutral tone  
            - When making technical statements:  
            – Preferably base them on ≥ 2 sources  
            – If not: soften with phrases like “in many cases,” “typically,” “most often”

            ---

            # ======================================
            # 9. OUTPUT FORMAT (P1)
            # ======================================

            • Allowed tags: `<h1>`, `<h2>`, `<p>`, `<ul>`, `<li>`, `<br>`, `<!--sec:slug-->`
            • Slug format: `^[a-z0-9-]+$`
            • Each <h2> section begins with a clear, independent opening sentence – no recourse to previously used formulations or statements, or repetition from other chapters.

            ### Structure:

            1. 1. `<h1>` with {page_title} + meaningful slug should be within the first slug if it is a banner.
            2. Optional: `<!--sec:banner-->`
            3. When specifying {SEC_MARKER_FMT}, the slug is the section name mentioned in the section lines or the rules (slug names are usually used like banner, (left_image_right_content, left_content_right_image, left_image_right_content2, left_content_right_image, FAQ, read more, etc.)
            4. For each line in SECTION_LINES:
            `<!--sec:slug(section_name)--><h2>…</h2>` with 150–200 words of text
            5. Optional: `<!--sec:sources--><h2>Sources</h2><ul><li>…</li></ul>`

            **Paragraph rule:**

            1. Write exactly one slug before each section HTML comment {SEC_MARKER_FMT.format(slug='section_name_or_the_slug(eg:banner,left_image_right_content,left_content_right_image)')} with the appropriate slug from the list below. Insert a visible <h2> heading immediately after it (except for the H1 at the beginning).

            # 10. QUALITY STANDARDS (P1–P3)

            • Each section addresses a **unique, clearly defined** topic  
            • No repetition in arguments, examples, or structure  
            • Sentence openings, lengths, and perspectives vary  
            • Paragraph transitions are smooth  
            • Style is factual, helpful, and grounded – no marketing speak  
            • Benefits are specific and credible  
            • No robotic syntax or pattern loops  
            • Maintain a **natural flow** with realistic tone variations
            ---
            # 11. LANGUAGE
            **Write the entire content in English language.**
        """
    else:
        pass    
    return prompt.strip()




def generate_full_page_text(page_title: str, keywords: str, sections_rules: List[Tuple[str, str, str]], language:str) -> str:
    prompt = build_longform_prompt(page_title, keywords, sections_rules, language)
    messages=""
    if language == 'deutsch':
        messages = [
            {"role": "system", "content": "Du bist ein professioneller deutschsprachiger Contentwriter mit Spezialisierung auf hochwertige und natürliche Servicetexte für Unternehmen. Deine Texte sind inhaltlich fundiert, klar strukturiert und stilistisch glaubwürdig. Du schreibst mit Fokus auf Relevanz, Verständlichkeit und sprachlicher Authentizität – ohne Floskeln, PR-Sprache oder künstliche Werbesignale. Dein Stil ist sachlich-empathisch, ruhig und informierend. Du sprichst Leser:innen auf Augenhöhe an, vermeidest dramatisierende Sprache und hältst dich an realistische Formulierungen mit erkennbarem Alltagsbezug. Du arbeitest mit natürlichem, variierendem Satzrhythmus und baust Absätze logisch und flüssig aufeinander auf. Jeder Text soll konsistent, klar gegliedert und in sich schlüssig sein – ohne isolierte Absätze oder wiederholte Einleitungen. Dein Ziel ist ein nutzerzentrierter, glaubwürdiger Text mit echtem Mehrwert – informierend, praxisnah und professionell zurückhaltend. Du beginnst jeden Abschnitt eigenständig, abwechslungsreich und thematisch klar und vermeidest dabei gleichförmige Einstiege."},
            {"role": "user", "content": prompt},
        ]
    elif language == 'englisch':
        messages = [
            {"role": "system", "content": "You are a professional english-speaking content writer specializing in high-quality, natural service texts for companies. Your texts are well-founded in content, clearly structured, and stylistically credible. You write with a focus on relevance, comprehensibility, and linguistic authenticity – without clichés, PR jargon, or artificial advertising signals. Your style is factual and empathetic, calm, and informative. You address readers at eye level, avoid overly dramatic language, and stick to realistic formulations with recognizable everyday relevance. You work with a natural, varied sentence rhythm and build paragraphs logically and fluently. Each text should be consistent, clearly structured, and coherent – without isolated paragraphs or repeated introductions. Your goal is a user-centered, credible text with real added value – informative, practical, and professionally restrained. You begin each section independently, varied, and thematically clear, avoiding monotonous introductions."},
            {"role": "user", "content": prompt},
        ]
    else:
        pass    
    # print('****************************************************************************************')
    # print(prompt)
    # print('****************************************************************************************')
    return chat_complete(messages, model="gpt-4o", temperature=0.8, top_p=1.0,
                         frequency_penalty=0.15, presence_penalty=0.25)


SECTION_RE = re.compile(
    r"<!--sec:(?P<slug>[a-z0-9_\-]+)-->[\s\r\n]*(?P<html>.*?)(?=(?:<!--sec:[a-z0-9_\-]+-->)|\Z)",
    re.I | re.S,
)


def split_longform_into_sections(full_html: str) -> Dict[str, Dict[str, str]]:
    """Parse the full HTML page into a dict keyed by slug.
    Returns
    -------
    dict
        slug -> {"title": <heading text>, "description": <full html chunk>}
    """
    
    sections: Dict[str, Dict[str, str]] = {}
    for match in SECTION_RE.finditer(full_html):
        slug = match.group('slug').lower()
        html_chunk = match.group('html').strip()
        # Extract first visible heading (h2/h3)
        t = re.search(r'<h[123]>(.*?)</h[123]>', html_chunk, re.I | re.S)
        # title = re.sub(r'<.*?>', '', t.group(1)).strip() if t else ''
        title = t.group(0).strip() if t else ''
        # Remove the heading from the description
        html_body = re.sub(r'^<h[123]>.*?</h[123]>', '', html_chunk, flags=re.I | re.S).strip()
        sections[slug] = {"title": title, "description": html_body}
    return sections



# QUALITY MEASURES / WORD COUNT
TAG_RE = re.compile(r'<[^>]+>')
WORD_RE = re.compile(r'\w+', re.UNICODE)


def strip_tags(html_text: str) -> str:
    """Remove HTML tags for word counting."""
    return TAG_RE.sub(' ', html_text or '')

 
def count_words(html_text: str) -> int:
    return len(WORD_RE.findall(strip_tags(html_text)))


def regen_short_section(page_title: str, slug: str, min_words: int, full_html: str, section_rule: str, language: str) -> str:
    regen_prompt = f"""
        Here is the full current HTML page content for the site "{page_title}":
        ```
        {full_html}
        ```
        The section marked with <!--sec:{slug}--> is too short.
        Please rewrite ONLY that section, keeping tone/style consistent, expanding it to at least {min_words} words (HTML excluded from count).
        **Follow this specific section rule when rewriting:**  
        "{section_rule}"
        Return ONLY the HTML for that section, starting with the <h2> heading (do NOT repeat the marker comment). if it's banner and bullet points are mentioned then add that also.
        write the content in {language} language.
        """
    messages = [
        {"role": "system", "content": f"You are a professional {language} web & SEO copywriter."},
        {"role": "user", "content": regen_prompt},
    ]
    new_html = chat_complete(messages, model="gpt-4o", temperature=0.7, top_p=1.0,
                             frequency_penalty=0.1, presence_penalty=0.2)
    return new_html


def build_output_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    topic_cols = ['Topic', 'Keywords'] + (['Number of Words'] if 'Number of Words' in input_df.columns else [])
    topics_df = input_df[topic_cols].dropna(subset=['Topic']).drop_duplicates('Topic')
    sections_df = input_df[['Sections Name']].dropna()
    sections = sections_df['Sections Name'].tolist()
    rows = []
    for _, row in topics_df.iterrows():
        topic = str(row['Topic']).strip()
        kw = str(row['Keywords']).strip()
        slug = normalize_slug(topic)
        first = True
        for sec in sections:
            rows.append({
                'type': 1 if first else 0,
                'page_title': topic if first else '',
                'page_slug': slug if first else '',
                'template': 'import' if first else '',
                'sections': sec,
                'keyword': kw,
                'title': '',
                'description': '',
                'image': '',
                'header_menu': 'home' if first else '',
                'meta_title': '',
                'meta_desc': '',
                'meta_image': ''
            })
            first = False
    df = pd.DataFrame(rows)    
    # df.to_csv('output_sheet.csv', index=False) 
    print('Raw output file created......')
    return df


def clean_description(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split('\n')
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = '\n'.join(lines)
    return text.strip()


def map_sections_to_df(df: pd.DataFrame, page_title: str, section_map: Dict[str, Dict[str, str]]):
    page_mask = df['page_title'] == page_title
    section_counter = defaultdict(int)
    for idx, row in df.loc[page_mask].iterrows():
        base_slug = normalize_slug(row['sections'])
        section_counter[base_slug] += 1
        slug = base_slug if section_counter[base_slug] == 1 else f"{base_slug}_{section_counter[base_slug]}"
        data = section_map.get(slug) or section_map.get(base_slug)
        if data:
            df.at[idx, 'title'] = data['title']
            df.at[idx, 'description'] = clean_description(data['description'])
        else:
            df.at[idx, 'title'] = ''
            df.at[idx, 'description'] = ''


def form_load(request):
    return render(request, 'form.html')


@csrf_exempt
def generate_response(request):

    if request.method != 'POST':
        return JsonResponse({'response': 'Invalid request'}, status=400)
    try:
        start_time = datetime.now()
        input_file = request.FILES.get('fileInput')
        language = request.POST.get("lanopt", "")
        print('Selected Language :',language)
        if input_file is None:
            return JsonResponse({'response': 'No file uploaded'}, status=400)
        input_df = pd.read_excel(input_file)
        required_cols = {'Topic', 'Keywords', 'Sections Name', 'Sections Rules','Number of words'}
        missing = required_cols.difference(input_df.columns)
        if missing:
            return JsonResponse({'response': f'Missing columns: {sorted(missing)}'}, status=400)
        
        df = build_output_dataframe(input_df)
        # Section rules dict (slug -> rule)
        rules: Dict[str, str] = {}
        for sec_name, sec_rule in zip(input_df['Sections Name'], input_df['Sections Rules']):
            rules[normalize_slug(sec_name)] = str(sec_rule).strip()
        # Generate per-topic longform, split, map, meta
        df["page_title"] = df["page_title"].replace('', pd.NA).ffill()
        grouped = df.groupby('page_title', sort=False)
        section_rows = input_df.dropna(subset=['Sections Name', 'Sections Rules'])
        for page_title, page_df in grouped:
            print(f"Creating content for page -> {page_title}")
            kw = page_df.iloc[0]['keyword']
            section_counter = defaultdict(int)
            sections_rules = []
            for _, row in section_rows.iterrows():
                section_name = str(row['Sections Name']).strip()
                section_rule = str(row['Sections Rules']).strip()
                word_count_str = str(row.get('Number of words', '')).strip()
                try:
                    word_count = int(word_count_str)
                except (ValueError, TypeError):
                    word_count = 100  # fallback default
                base_slug = normalize_slug(section_name)
                section_counter[base_slug] += 1
                final_slug = base_slug if section_counter[base_slug] == 1 else f"{base_slug}_{section_counter[base_slug]}"
                sections_rules.append((final_slug,word_count, section_rule))
            full_html = generate_full_page_text(page_title, kw, sections_rules,language)

            # Split
            section_map = split_longform_into_sections(full_html)
            # Step 1: Build a quick lookup for expected word counts per slug
            slug_to_wordcount = {slug: word_count for slug, word_count, _ in sections_rules}
            # print('******')
            # print(section_map.items())
            # print('*******')
            # Step 2: Check and regenerate short sections
            for slug, data in list(section_map.items()):
                target_words = slug_to_wordcount.get(slug, 150)  # fallback to 150 if not found
                actual_words = count_words(data['description'])

                if actual_words < target_words * WORD_TOLERANCE:
                    # Attempt targeted regeneration
                    print(f"Content is short so regenerating section {slug} again..")
                    slug_to_rule = {slug: rule for slug, _, rule in sections_rules}
                    section_rule = slug_to_rule.get(slug, "")  # fallback to empty string if not found
                    new_html = regen_short_section(page_title, slug, target_words, full_html,section_rule,language)
                    if new_html:
                        # Extract new title from <h[123]>
                        t = re.search(r'<h[123]>.*?</h[123]>', new_html, re.I | re.S)
                        if t:
                            # title = re.sub(r'<.*?>', '', t.group(0)).strip()
                            title = t.group(0).strip()
                            description = new_html.replace(t.group(0), '', 1).strip()
                        else:
                            title = ''
                            description = new_html.strip()
                            
                        # description = re.sub(r'^```html|^```|```$', '', description).strip()
                        section_map[slug]['title'] = title
                        section_map[slug]['description'] = description
                    # if new_html:
                    #     section_map[slug]['description'] = new_html
                    #     # Refresh title
                    #     t = re.search(r'<h[123]>(.*?)</h[123]>', new_html, re.I | re.S)
                    #     if t:
                    #         section_map[slug]['title'] = re.sub(r'<.*?>', '', t.group(1)).strip()

            # Map into main DF
            map_sections_to_df(df, page_title, section_map)
            first_idx = page_df.index[0]
            if df.loc[first_idx, 'type'] == 1:
                messages = f"""Erstelle eine einsatzbereite Meta-Title für '{page_title}'
                                Format:
                                Exakt 40-44 Zeichen inkl. Leerzeichen (Obligatorisch)
                                Genau ein Emoji in der Mitte (aus 4 passenden Optionen wählen, dann 1 zufällig einsetzen)
                                Fokus-Keyword möglichst vorn
                                Kein Firmenname, kein Wortspiel zu sensiblen Themen.In {language}"""
                meta_title = chat_complete([{"role": "system", "content": f"You are a professional {language} SEO writer."},
                                            {"role": "user", "content": messages}
                                            ], model="gpt-4o", temperature=0.6)
                df.loc[first_idx, "meta_title"] = meta_title
                messages = f"""Meta-Beschreibung (normal formatiert)
                                    Exakt 120–150 Zeichen inkl. Leerzeichen (Obligatorisch)
                                    Mit topic '{page_title}', 1–2 Nebenkeywords, klarem USP & sympathischem CTA
                                    Ton: positiv, respektvoll, inklusiv, mit subtil charmantem Stil
                                    Keine plumpen Werbephrasen, kein Pathos bei sensiblen Themen.In {language}"""
                meta_desc = chat_complete([{"role": "system", "content": f"You are a professional {language} SEO writer."},
                                            {"role": "user", "content": messages}
                                            ], model="gpt-4o", temperature=0.6)
                df.loc[first_idx, "meta_desc"] = meta_desc
            
            
        # Cleanup columns not needed downstream
        df.loc[df['type'] == 0, 'page_title'] = ""
        if 'keyword' in df.columns:
            df.drop(columns=['keyword'], inplace=True)
        # Export CSV
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_b64 = base64.b64encode(buf.getvalue().encode('utf-8')).decode('utf-8')

        elapsed = datetime.now() - start_time
        print('Generation finished in', elapsed)
        #Calculate cost
        input_cost = (TOTAL_PROMPT_TOKENS / 1_000_000) * GPT4O_INPUT_COST_PER_M
        output_cost = (TOTAL_COMPLETION_TOKENS / 1_000_000) * GPT4O_OUTPUT_COST_PER_M
        total_cost = input_cost+output_cost
        print(f"Token usage: input={TOTAL_PROMPT_TOKENS}, output={TOTAL_COMPLETION_TOKENS}, total={TOTAL_TOKENS}")
        print(f"Estimated cost: ${total_cost:.4f}")
        return JsonResponse({
            'csv_base64': csv_b64,
            'message': 'Generation completed successfully.',
            'token_usage': {
                'total_tokens': TOTAL_TOKENS,
                'prompt_tokens': TOTAL_PROMPT_TOKENS,
                'completion_tokens': TOTAL_COMPLETION_TOKENS,
                'cost':total_cost,
                
            },
        })

    except Exception as exc:  #
        print('Exception occurred:', exc)
        return JsonResponse({'response': 'Error occurred'}, status=500)


