import requests
import json

class PerplexityDeepSearch:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.search_history = []

    def analyze_text(self, news_text):
        """Initial context injection with news text"""
        payload = {
            "model": "sonar-reasoning-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant. Analyze and remember this news text: " + news_text
                },
                {
                    "role": "user",
                    "content": "Confirm understanding of the provided news text and prepare for follow-up questions."
                }
            ],
            "web_search_options": {
                "search_context_size": "high"
            }
        }
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        return response.json()

    def ask_question(self, question):
        """Query with reasoning steps and search history"""
        payload = {
            "model": "sonar-reasoning-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Provide detailed reasoning steps and search queries used."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "web_search_options": {
                "search_context_size": "high"
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "reasoning_steps": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "search_queries": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
        
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        result = response.json()
        self.search_history.append({
            "question": question,
            "response": result
        })
        return result


# usage Example
API_KEY = open('PerplexityAI-API-key').read().strip()

# news text
news_text = "KONTROLLUTVALGET Til Kommunestyret i Tromsø kommune KONTROLLUTVALGETS UTTALELSE OM TROMSBADET KFs ÅRSREGNSKAP OG ÅRSBERETNING FOR 2023 Kontrollutvalget har i møte 30.5.2024 behandlet Tromsøbadet KFs årsregnskap og årsberetning for 2023. Grunnlaget for behandlingen har vært årsregnskapet til Tromsøbadet KF, styrets årsberetning og revisjonsberetningen datert. 15.04.2024. I tillegg har revisor supplert kontrollutvalget med muntlig informasjon om aktuelle problemstillinger. Kontrollutvalget har merket seg at årsregnskapet til Tromsøbadet KF for 2023 er gjort opp i balanse og viser et positivt netto driftsresultat på kr 812 124. Investeringsregnskapet for 2023 er gjort opp med et udekket beløp på kr. 506 816 som er fremført til inndekning i senere år. Årsaken er knyttet til investeringer i varmeanlegg og hinderløyper som ikke var budsjettert med i årets investeringsbudsjett. Foretakets økonomiske resultat, tallstørrelser og regnskapsposter er kommentert nærmere i årsberetningen og det vises til den. Kontrollutvalget har også merket seg at årsregnskapet og årsberetningen er avlagt uten vesentlige feil, og at revisjonsberetningen er uten merknader. Gjennom rapporteringer fra revisor har ikke kontrollutvalget blitt gjort oppmerksom på forhold som skal tilsi at revisjonen ikke har blitt utført på en betryggende måte. Kontrollutvalget anbefaler at kommunestyret godkjenner Tromsøbadet KFs årsregnskap og årsberetning for 2023. Tromsø, den 30.mai 2024 Line Fusdahl leder av kontrollutvalget Gjenpart: Formannskapet Styret i Tromsøbadet KF"
questions = [
    'Are local residents or groups actively involved in this event?',
    'Is the event part of a sustained campaign or movement?',
    'Does it demonstrate community agency or collective action?',
    'Could it inspire further participation or engagement?',
    'Does it reflect democratic or civic values locally?',
]

pds = PerplexityDeepSearch(API_KEY)
analysis_response = pds.analyze_text(news_text)
print(json.dumps(analysis_response, indent=2))

question_responses = []
for question in questions:
    response = pds.ask_question(question)
    print(json.dumps(response, indent=2))
    question_responses.append(response)
    
OUTPUT_FILE = 'test_output.json'
with open(OUTPUT_FILE, 'w') as f:
    json.dump(analysis_response, f, indent=2)
    json.dump(question_responses, f, indent=2)
print(f"Responses saved to {OUTPUT_FILE}")
