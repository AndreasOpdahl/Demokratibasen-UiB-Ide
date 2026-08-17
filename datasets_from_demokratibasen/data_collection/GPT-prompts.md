# Prompts Used by Demokratibasen for GPT inferencing

## Models

* GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")  # until 2025-09-03, gpt-4-turbo-preview may also have been used...
* GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # after 2025-09-03

## Output tokens

* MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", 1000))  # since 2024-08-29
* Apparently no limits before that

## Instructions

### Since 2024-08-29

INSTRUCTIONS = """

    Svar på følgende JSON format: {

    "summary_title": "Du er en politisk journalist, lag en
overskrift som passer til innholdet i dokumentet og oppsummeringen.
Unngå å bruke floskler og klisjéer, som for eksempel 'viktig'. Bruk
norske tittelregler, kun stor forbokstav i første ord.",

    "summary_body": "Du er en politisk journalist, skriv en
kortfattet oppsummering av dokumentet.",

    "keywords": ["Liste over sentrale stikkord i dokumentet"],

    "persons_mentioned": ["Liste med navn på personene nevnt i
dokumentet"],

    "news_score": "Du er en politisk journalist. Ut ifra innholdet
i dokumentet, gi en vurdering på en skala fra 0 til 100 på dokumentets
nyhetsverdi.

    Saker med stor innvirkning som påvirker mange mennesker på
ulike vis bør gi en høy score,

    mens kjedelige dokumenter der sakene er lav offentlig interesse
bør gi lav score.

    ",

}"""

### Before 2024-08-29

Before 2024-08-29, there was an even simpler prompt, I think with no MAX_TOKENS.

    instructions = """{

    "summary_body": "En kortfattet oppsummering av dokumentet.",

    "summary_title": "En overskrift som passer til innholdet i
dokumentet og oppsummeringen.",

    "journalistic_stories": [

    {

    "title": "Tittel på en mulig nyhetsartikkel om
dokumentet.",

    "body": "To til tre setninger om hva nyhetsartikkelen
skal handle om.",

    }

    ],

    "stated_interests": "En beskrivelse av relevante egeninteresser
i dokumentet.",

    "keywords": ["Liste over sentrale stikkord i dokumentet"],

    "persons_mentioned": ["Liste med navn på personene nevnt i
dokumentet"],

    "news_score": "

    Vurdering på en skala fra 0 til 100 av sjansen for at
dokumentet vil kunne gi en nyhetsverdig artikkel.

    Store og komplekse dokumenter som angår mange personer og  i
kommunen bør gi høy score,

    mens korte og enkle dokumenter som omhandler interne forhold
bør gi lav score

    ",

    }"""
