# TP1 — Premiers pas avec la génération de texte (≈ 1h30)

[⬅️ Retour au sommaire](../LISEZMOI.md)

## Objectifs

- Comprendre les composants essentiels de LangChain et leur intérêt pour l'orchestration de modèles de langage (LLM).
- Créer et configurer un compte Mistral AI afin d'obtenir une clé API personnelle.
- Mettre en place un environnement Python isolé pour expérimenter avec LangChain et Mistral.
- Réaliser vos premières invocations d'un modèle Mistral via LangChain, puis industrialiser cette interaction à l'aide de prompts structurés et de chaînes.

## Prérequis

- Python 3.10+ installé localement.
- Un éditeur de code (VS Code, PyCharm, etc.).
- Connaissances de base en Python (gestion des paquets, exécution de scripts).

## Ressources utiles

- **LangChain** : [Documentation](https://python.langchain.com/docs/introduction/) · [Chaîne YouTube](https://www.youtube.com/@LangChain)
- **Mistral AI** : [Documentation officielle](https://docs.mistral.ai/)
- **OpenAI** (référence optionnelle) : [Documentation](https://platform.openai.com)

## Étape 0 — Création du compte Mistral AI (≈ 10 min)

1. Rendez-vous sur [https://mistral.ai/](https://mistral.ai/).
2. Cliquez sur **Try the API** puis **S'inscrire**.
3. Choisissez l'abonnement **Gratuit / expérimental**.
4. Dans l'onglet **API → Clés API**, créez une nouvelle clé.
5. Copiez cette clé : vous en aurez besoin dans vos scripts Python (ne la partagez pas).

> 💡 Vous pouvez aussi créer une clé de test OpenAI pour comparer les comportements des modèles. Cela reste optionnel.

## Étape 1 — Préparer l'environnement Python (≈ 10 min)

1. Créez un dossier de travail pour ce TP et ouvrez un terminal dans ce dossier.
2. Créez un environnement virtuel :
   ```bash
   python -m venv mon_env
   source mon_env/bin/activate  # Sous Windows : mon_env\Scripts\activate
   ```
3. Installez les bibliothèques nécessaires :
   ```bash
   pip install langchain langchain-mistralai langchain-openai python-dotenv
   ```
4. (Optionnel) Placez vos clés dans un fichier `.env` :
   ```env
   MISTRAL_API_KEY="votre-cle"
   OPENAI_API_KEY="votre-cle-openai"  # Optionnel
   ```
5. Désactivez l'environnement via `deactivate` quand vous avez terminé.

> 💡 `python-dotenv` permet de charger vos variables d'environnement sans les écrire dans le code. Vous pouvez aussi définir `export MISTRAL_API_KEY=...` directement dans le terminal.

## Étape 2 — Comprendre les notions de LangChain (≈ 10 min)

LangChain est un framework **agnostique** vis-à-vis des modèles qui facilite :

- l'invocation de différents LLM (Mistral, OpenAI, etc.) ;
- l'automatisation des prompts via des **chaînes** ;
- la gestion d'**agents** (raisonnement multi-étapes) ;
- l'intégration de **vector stores** et d'**embeddings** pour les applications RAG ;
- la construction d'architectures avancées (LangGraph, multi-agents...).

👉 **Exercice 2.1** — En quelques phrases, décrivez un cas d'usage concret (professionnel ou personnel) qui pourrait tirer parti de LangChain.

👉 **Exercice 2.2** — Identifiez, dans la documentation officielle, un exemple de chaîne ou d'agent que vous souhaiteriez tester plus tard. Indiquez la page ou la ressource correspondante.

## Étape 3 — Première invocation Mistral (≈ 20 min)

Créez un script `tp1_mistral.py` avec le contenu suivant :

```python
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    # api_key="..."  # facultatif si vous utilisez la variable d'environnement
)

message = HumanMessage(content="Quelle est la capitale de l'Albanie ?")
response = llm.invoke([message])

print(response.content)
```

1. Ajoutez l'import `import os` et utilisez `os.environ["MISTRAL_API_KEY"]` (ou `os.getenv`) pour sécuriser la récupération de la clé.
2. Exécutez le script et vérifiez la réponse.
3. Remplacez la question par deux requêtes différentes pour observer la cohérence des sorties.

👉 **Exercice 3.1** — En adaptant le script, affichez également `response.usage_metadata` afin d'inspecter le nombre de tokens consommés. Calculez ensuite un coût approximatif en vous basant sur le tarif public (`input_tokens` et `output_tokens`).

👉 **Exercice 3.2** — Modifiez la température (`temperature=0.7`) et observez les différences de réponses pour une requête créative.

> 💡 Pour comparer, vous pouvez également invoquer `ChatOpenAI` (modèle `gpt-4.1`) en adaptant deux lignes de code.

## Étape 4 — Chaînes et prompts structurés (≈ 25 min)

LangChain facilite la composition d'étapes (prompt → modèle → parser). Reprenez votre script dans un nouveau fichier `tp1_chain.py` :

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

model = ChatMistralAI(model="mistral-large-latest", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Fais-moi une blague sur le sujet : {sujet}"
)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

for sujet in ["pompier", "police"]:
    print(chain.invoke({"sujet": sujet}))
    print("-" * 10)
```

👉 **Exercice 4.1** — Remplacez la liste de sujets par une liste saisie dynamiquement (prompt utilisateur ou lecture d'un fichier). Ajoutez un contrôle qui empêche l'envoi d'une chaîne vide.

👉 **Exercice 4.2** — Créez une variante de la chaîne produisant des réponses au format JSON (par exemple avec les clés `setup` et `punchline`). Pour cela, adaptez le prompt et utilisez `StrOutputParser()` ou un parser JSON.

👉 **Exercice 4.3** — Ajoutez un paramètre `temperature` passé dynamiquement via `chain.invoke` pour comparer l'impact sur plusieurs sujets.

## Étape 5 — Prompt hiérarchisé et métadonnées (≈ 15 min)

Les prompts peuvent combiner un message système et un message utilisateur :

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "Vous êtes un rédacteur de documentation technique de classe mondiale."),
    ("user", "{input}")
])

llm = ChatMistralAI(model="mistral-large-latest")
chain = prompt | llm

result = chain.invoke({"input": "Qu'est-ce que le modèle mistral-large-latest ?"})
print(result.content)
print(result.usage_metadata)
```

👉 **Exercice 5.1** — Modifiez le message système pour orienter le style (ex. ton humoristique, réponse concise, etc.). Testez plusieurs variantes et mesurez à chaque fois l'impact sur `usage_metadata`.

👉 **Exercice 5.2** — À partir de `usage_metadata`, estimez le coût de trois requêtes différentes. Présentez vos calculs dans un tableau synthétique.

👉 **Exercice 5.3** *(optionnel)* — Inspirez-vous de la documentation LangChain pour ajouter un `StrOutputParser` et ne récupérer que du texte brut, puis enregistrez la sortie dans un fichier `.md`.

## Pour aller plus loin

- Explorez les **vector stores** et les embeddings pour mettre en place un mini-système RAG (Retrieval Augmented Generation).
- Testez les **LangGraphs** ou les **agents** pour orchestrer plusieurs étapes de raisonnement.
- Comparez les performances et coûts entre Mistral et un modèle OpenAI sur une tâche donnée.

## Rendu attendu

- Scripts Python (`tp1_mistral.py`, `tp1_chain.py`, variantes éventuelles).
- Un court compte rendu (1 page max) résumant :
  - votre compréhension de LangChain et Mistral ;
  - les expérimentations menées ;
  - les coûts observés ;
  - les axes d'amélioration ou d'applications futures.

Bon TP !
