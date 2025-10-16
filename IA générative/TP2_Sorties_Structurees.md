# TP2 — Sorties structurées avec LangChain et Mistral (≈ 1h30)

[⬅️ Retour au sommaire](../LISEZMOI.md)

## Objectifs

- Comprendre l'intérêt des **sorties structurées** pour fiabiliser les réponses d'un LLM.
- Mettre en place des schémas Pydantic et les utiliser avec `with_structured_output`.
- Explorer plusieurs formats de sortie : booléen, classe, entier, traduction, raisonnement pas à pas.
- Industrialiser les résultats (post-traitement, extraction d'informations) au sein d'une chaîne LangChain.

## Prérequis

- Avoir réalisé (ou parcouru) le [TP1 — Premiers pas avec la génération de texte](TP1_LangChain_Mistral.md).
- Python ≥ 3.10, un environnement virtuel actif et la bibliothèque `langchain-mistralai` installée.
- Une clé API Mistral AI valide placée dans vos variables d'environnement.

## Étape 1 — Pourquoi structurer les sorties ? (≈ 10 min)

Sans contrainte, un modèle peut répondre dans un format imprévisible. Les **sorties structurées** permettent :

- d'imposer un schéma de validation (types, valeurs minimales / maximales, listes d'options) ;
- d'éviter des parsings fragiles (regex, JSON non valide, etc.) ;
- de faciliter la consommation des résultats dans vos applications.

👉 **Exercice 1.1** — Listez deux scénarios métiers où recevoir une sortie strictement structurée est indispensable.

👉 **Exercice 1.2** — Dans votre propre expérience, quels problèmes avez-vous déjà rencontrés à cause de sorties libres de LLM ?

## Étape 2 — Cas simple : réponse booléenne (≈ 15 min)

Créez un fichier `tp2_bool.py` avec le contenu suivant :

```python
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

class Answer(BaseModel):
    answer: bool

prompt_answer = [
    ("system", "Tu es un assistant chargé de répondre un booléen (True ou False) à la question d'un utilisateur."),
    ("human", "{question}")
]

prompt_answer_template = ChatPromptTemplate.from_messages(prompt_answer)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt_answer_template | llm.with_structured_output(schema=Answer)

def repond(question):
    return chain.invoke({"question": question}).answer

for question in ["Noël est en hiver", "Il pleut quand il pleut pas"]:
    print(question)
    reponse = repond(question)
    print(f"Réponse : {reponse} (type : {type(reponse)})")
    print()
```

👉 **Exercice 2.1** — Ajoutez un troisième exemple de question ambigüe. Observez la sortie et discutez de la façon dont le prompt pourrait être renforcé.

👉 **Exercice 2.2** — Vérifiez le type Python retourné. Pourquoi est-ce important pour la suite de votre application ?

## Étape 3 — Choisir une action prédéfinie (≈ 20 min)

Nous pouvons imposer une liste d'actions possibles. Créez `tp2_action.py` :

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

tasks = ["Répondre à une nouvelle question", "Fournir plus d'éléments à la question précédente"]

class NextTask(BaseModel):
    """Utilise toujours cet outil pour structurer ta réponse to the user."""
    action: str = Field(
        ...,
        enum=tasks,
        description="La prochaine action à mener"
    )

prompt_message = [
    ("system", "Tu es un assistant chargé de classifier la demande d'un utilisateur parmi une liste réduite d'actions à mener en tant que chatbot. Tu dois déterminer la prochaine action à mener."),
    ("human", "{text}")
]

prompt = ChatPromptTemplate.from_messages(prompt_message)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt | llm.with_structured_output(schema=NextTask)

for text in ["Peux-tu m'en dire plus", "Que sont les PPV ?"]:
    print(text)
    print(chain.invoke({"text": text}))
    print()
```

👉 **Exercice 3.1** — Ajoutez une troisième action (« Demander une clarification ») et vérifiez que le LLM la sélectionne lorsqu'un utilisateur est confus.

👉 **Exercice 3.2** — Comparez la robustesse de ce schéma structuré à une simple réponse textuelle : que se passe-t-il si le modèle hallucine ?

## Étape 4 — Quantifier un comportement : note entière (≈ 15 min)

Passons à un score numérique. Créez `tp2_ton.py` :

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

class TonMessage(BaseModel):
    """Évaluation du ton du message de l'utilisateur."""
    note_ton: int = Field(
        ...,
        ge=1,
        le=5,
        description="Note attribuée au ton du message : 1 pour neutre, 5 pour très aimable"
    )

prompt_message = [
    ("system", "Tu es un assistant chargé d'évaluer le ton d'un message donné par l'utilisateur. Attribue une note de 1 à 5 au ton du message, où 1 signifie neutre et 5 signifie très aimable."),
    ("human", "{text}")
]

prompt = ChatPromptTemplate.from_messages(prompt_message)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt | llm.with_structured_output(schema=TonMessage)

messages = [
    "Bonjour, pourrais-tu m'aider s'il te plaît ?",
    "J'ai besoin de ça immédiatement.",
    "Merci beaucoup pour ton aide précieuse !"
]

for text in messages:
    print(f"Message : {text}")
    print(chain.invoke({"text": text}))
    print()
```

👉 **Exercice 4.1** — Ajoutez un champ `commentaire: str` pour expliquer la note. Quelle consigne ajouter dans le prompt système ?

👉 **Exercice 4.2** — Transformez ce score en métrique agrégée : calculez la moyenne de `note_ton` sur une liste de messages et affichez le résultat.

## Étape 5 — Forcer un format JSON exploitable (≈ 20 min)

Même avec un prompt clair, le modèle peut répondre librement :

```python
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage

llm = ChatMistralAI(model="mistral-medium-latest")
message = HumanMessage(content="Peux-tu me traduire ce qui suit, en anglais ?\n\n Quelle est la capitale de l'Albanie ?")
print(llm.invoke([message]))
```

Pour garantir un JSON valide, imposez un schéma `Translation` :

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel, Field

class Translation(BaseModel):
    original_text: str = Field(..., description="The original text before translation in another language")
    original_language: str = Field(..., description="The original language before translation")
    translated_text: str = Field(..., description="The final text after translation in another language")
    translated_language: str = Field(..., description="The language into which the translation must be done")

def traduit(texte, langue_source="français", langue_cible="anglais"):
    llm = ChatMistralAI(model_name="mistral-medium-latest")
    prompt = ChatPromptTemplate.from_template("""Je souhaite que tu traduises le texte suivant du {langue_source} vers le {langue_cible}. Ta traduction doit être précise, fluide et naturelle, et préserver parfaitement le sens original. 
    Retourne-moi la réponse sous forme d'objet JSON avec les champs :
      - original_text : le texte original
      - original_language : la langue du texte original
      - translated_text : la traduction du texte
      - translated_language : la langue de la traduction

    Voici le texte à traduire :
    ----
    {texte}""")
    output_parser = StrOutputParser()
    extract_translation = RunnableLambda(lambda translation: translation.translated_text)
    chain0 = prompt | llm.with_structured_output(Translation) | extract_translation
    return chain0.invoke({
        "langue_source": langue_source,
        "langue_cible": langue_cible,
        "texte": texte
    })

print(traduit("Quelle est la capitale de l'Albanie"))
```

👉 **Exercice 5.1** — Supprimez temporairement `RunnableLambda`. Que récupérez-vous ? Comment pourriez-vous exploiter l'objet complet ?

👉 **Exercice 5.2** — Modifiez la fonction pour prendre en charge d'autres couples de langues. Ajoutez un contrôle qui lève une exception si `langue_cible` n'est pas supportée.

## Étape 6 — Structurer un raisonnement pas à pas (≈ 15 min)

Les sorties structurées peuvent guider un raisonnement élémentaire :

```python
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

class Etape(BaseModel):
    explication: str
    sortie: str

class MathReponse(BaseModel):
    etapes: list[Etape]
    reponse_finale: str

prompt_answer = [
    ("system", "Tu es un professeur de mathématiques très pédagogue."),
    ("human", "{exercice}")
]

prompt_answer_template = ChatPromptTemplate.from_messages(prompt_answer)
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
chain = prompt_answer_template | llm.with_structured_output(schema=MathReponse)

explications = chain.invoke({"exercice": "Résous  8x + 31 = 2"})
for etape in explications.etapes:
    print(f"- {etape.explication}")
    print(f"  Le résultat est alors : {etape.sortie}")

print(f"Au final, on trouve : {explications.reponse_finale}")
```

👉 **Exercice 6.1** — Ajoutez un champ `verifications: list[str]` pour forcer le modèle à contrôler sa solution.

👉 **Exercice 6.2** — Testez le prompt sur d'autres équations. Quelles limites identifiez-vous ? (par exemple, expressions non linéaires, fractions complexes...)

## Pour aller plus loin

- Explorez les agents LangChain capables de combiner plusieurs outils structurés.
- Connectez ces schémas à un stockage de données (base SQL, vector store) pour automatiser des workflows.
- Comparez le comportement avec d'autres fournisseurs (OpenAI, Anthropic) : les schémas Pydantic restent compatibles.

Bon TP !
