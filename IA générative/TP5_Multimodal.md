# TP5 — Multimodalité et IA générative (≈ 2h)

[⬅️ Retour au README](../README.md)

## Objectifs

- Comprendre ce que recouvre la multimodalité dans les API d'IA générative modernes.
- Mettre en place des pipelines audio (synthèse vocale ↔ transcription) avec l'API OpenAI.
- Exploiter la vision et la génération d'images pour enrichir une application.
- Composer plusieurs modalités dans un mini-projet intégrant texte, image et audio.

## Prérequis

- Python 3.10+ et un shell avec `ffmpeg` installé (requis par `pydub` pour l'audio).
- Une clé d'API OpenAI stockée dans `OPENAI_API_KEY` (ou un fichier `.env`).
- Avoir suivi les TP1 à TP4 pour être à l'aise avec les appels API et la structuration de projets.

## Ressources utiles

- [Documentation OpenAI Python](https://platform.openai.com/docs/guides/setup) — clients `responses`, `audio`, `images`.
- [Guide TTS (`audio.speech`)](https://platform.openai.com/docs/guides/text-to-speech).
- [Guide transcription (`audio.transcriptions`)](https://platform.openai.com/docs/guides/speech-to-text).
- [Référence multimodale (`responses`)](https://platform.openai.com/docs/guides/vision).
- [Pydub](https://github.com/jiaaro/pydub) pour manipuler l'audio côté Python.

---

## Étape 0 — Pourquoi la multimodalité ? (≈ 10 min)

La multimodalité permet à un modèle de comprendre ou de produire plusieurs types de données (texte, image, audio, vidéo). Les combinaisons les plus courantes en production restent **texte ↔ image** et **texte ↔ audio**.

- Les modèles *vision* interprètent des pixels pour répondre à un prompt textuel.
- Les modèles *speech* synthétisent ou transcrivent une voix.
- Les API modernes (OpenAI `responses`, `audio`) autorisent des entrées mixtes : texte + image + instructions, ou encore texte + audio.

👉 **Exercice 0.1** — Listez trois cas d’usage multimodaux dans votre contexte professionnel (support client, maintenance, médiation culturelle...).

👉 **Exercice 0.2** — Pour chacun, identifiez la combinaison de modalités (entrée → sortie) nécessaire.

---

## Étape 1 — Préparer l’environnement (≈ 15 min)

1. Créez un environnement isolé et installez les dépendances :
   ```bash
   python -m venv tp5_multimodal
   source tp5_multimodal/bin/activate  # Windows : tp5_multimodal\Scripts\activate
   pip install --upgrade openai python-dotenv pydub pillow rich
   ```
2. Ajoutez `ffmpeg` à votre PATH. Sous Ubuntu/Debian : `sudo apt install ffmpeg`. Sous Windows, installez la version officielle puis ajoutez le dossier `bin` à la variable d’environnement.
3. Créez un fichier `.env` à la racine de votre projet :
   ```env
   OPENAI_API_KEY="votre-cle"
   ```
4. Chargez la clé dans vos scripts :
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

👉 **Exercice 1.1** — Vérifiez que `import openai; from openai import OpenAI` s’exécutent sans erreur dans un REPL.

👉 **Exercice 1.2** — Ajoutez `rich` comme dépendance pour colorer vos logs et construisez un petit utilitaire `log.py` réutilisable dans tout le TP.

---

## Étape 2 — Du texte vers la parole (Text-to-Speech) (≈ 20 min)

L’API `audio.speech` synthétise une voix naturelle à partir d’un prompt textuel. Les modèles `gpt-4o-mini-tts` et `gpt-4o-audio-preview` offrent un bon compromis qualité/coût.

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI()

def synthesize_message(message: str, out_path: str, voice: str = "alloy") -> Path:
    """Génère un fichier MP3 à partir d'un texte."""
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=message,
    ) as response:
        response.stream_to_file(output)

    return output

if __name__ == "__main__":
    path = synthesize_message(
        "Bonjour ! Prêt à explorer la multimodalité avec OpenAI ?",
        "sorties/audio/introduction.mp3",
    )
    print(f"Fichier généré : {path.resolve()}")
```

👉 **Exercice 2.1** — Testez au moins trois voix (`alloy`, `verse`, `nova`, `solaria`...), comparez les timbres et notez vos préférences.

👉 **Exercice 2.2** — Ajoutez un paramètre `language` (par exemple `"fr-FR"`, `"en-US"`) pour piloter la prononciation et faites varier le contenu (dialogue, consignes, storytelling).

---

## Étape 3 — De l’audio vers le texte (Speech-to-Text) (≈ 25 min)

Pour transcrire des fichiers longs, découpez-les en segments manipulables. `pydub` gère la lecture/export, l’API OpenAI se charge de la transcription.

```python
from pathlib import Path
from typing import List
from openai import OpenAI
from pydub import AudioSegment
import tempfile

client = OpenAI()

def transcribe_mp3(mp3_path: str, chunk_minutes: int = 2) -> str:
    """Transcrit un MP3 long en effectuant un traitement par blocs."""
    audio = AudioSegment.from_file(mp3_path)
    chunk_ms = chunk_minutes * 60 * 1000
    transcripts: List[str] = []

    with tempfile.TemporaryDirectory() as workdir:
        for index, start in enumerate(range(0, len(audio), chunk_ms)):
            segment = audio[start:start + chunk_ms]
            if len(segment) < 1_000:  # ignore les fragments trop courts
                continue

            chunk_path = Path(workdir) / f"chunk_{index}.mp3"
            segment.export(chunk_path, format="mp3")

            with chunk_path.open("rb") as chunk_file:
                result = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=chunk_file,
                    response_format="verbose_json",
                )
            transcripts.append(result.text.strip())

    return " ".join(transcripts)

if __name__ == "__main__":
    texte = transcribe_mp3("donnees/reunion.mp3")
    Path("sorties/transcriptions").mkdir(parents=True, exist_ok=True)
    (Path("sorties/transcriptions") / "reunion.txt").write_text(texte, encoding="utf-8")
    print("Transcription terminée.")
```

👉 **Exercice 3.1** — Mesurez le temps de traitement par bloc (utilisez `time.perf_counter`). Quelle taille de segment optimise le rapport coût/délai ?

👉 **Exercice 3.2** — Expérimentez `response_format="json"` et parsez les horodatages (`segments`) pour reconstituer une timeline détaillée.

---

## Étape 4 — Interroger une image (Vision) (≈ 30 min)

Les appels `responses.create` acceptent des contenus mixtes. Ci-dessous, on convertit une image locale en data URL pour l’envoyer au modèle `gpt-4.1-mini`.

```python
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI

client = OpenAI()

def encode_image_to_data_url(image_path: str) -> str:
    """Encode une image locale en data URL (base64) prête pour l’API."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        raise ValueError(f"Impossible de déduire le MIME type de {path}")

    image_bytes = path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def describe_image(image_path: str, question: str) -> str:
    data_url = encode_image_to_data_url(image_path)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        temperature=0.1,
        max_output_tokens=300,
    )
    texts = []
    for item in response.output:
        for content in item.content:
            if content.type == "output_text":
                texts.append(content.text.strip())
    return "\n".join(texts)


if __name__ == "__main__":
    analyse = describe_image("img/tableau_de_bord.png", "Que peux-tu déduire de cette capture d'écran ?")
    print(analyse)
```

👉 **Exercice 4.1** — Remplacez l’image par une photo de document manuscrit. Comparez la qualité de l’abrégé produit par `gpt-4.1-mini` vs `gpt-4.1`.

👉 **Exercice 4.2** — Ajoutez une contrainte de formatage (par exemple : bullet points, JSON) et vérifiez la robustesse de la réponse.

---

## Étape 5 — Générer une image (≈ 20 min)

Complétez la boucle en produisant un visuel à partir d’un prompt textuel. L’API `images.generate` renvoie du base64 qu’il suffit de sauvegarder.

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI()

def generate_image(prompt: str, out_path: str, size: str = "1024x1024") -> Path:
    """Génère une image et la sauvegarde localement."""
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        quality="high",
    )
    image_b64 = result.data[0].b64_json
    output.write_bytes(base64.b64decode(image_b64))
    return output


if __name__ == "__main__":
    chemin = generate_image(
        "Poster minimaliste illustrant la rencontre entre l'audio et la vision en IA.",
        "sorties/images/multimodal.png",
    )
    print(f"Image générée : {chemin.resolve()}")
```

👉 **Exercice 5.1** — Paramétrez `size` (`512x512`, `1792x1024`...) et `quality` (`standard`, `high`). Observez l’impact sur le rendu et le temps de génération.

👉 **Exercice 5.2** — Créez trois prompts différents puis réalisez un vote en binôme pour sélectionner la meilleure image.

---

## Étape 6 — Composer un pipeline multimodal (≈ 30 min)

Assemblez les briques audio + vision pour livrer une expérience complète. Commencez par factoriser vos fonctions dans deux modules (`pipeline_audio.py` pour `synthesize_message`, `pipeline_vision.py` pour `describe_image`). Exemple : analyser une infographie, résumer en texte puis générer un audio de synthèse.

```python
from pathlib import Path
from pipeline_audio import synthesize_message
from pipeline_vision import describe_image

def analyse_visuelle_vers_audio(image_path: str, question: str) -> Path:
    """Chaîne complète vision → texte → audio."""
    resume = describe_image(image_path, question)
    print("Résumé généré :\n", resume)
    sortie_audio = synthesize_message(
        message=f"Voici mon analyse : {resume}",
        out_path="sorties/audio/analyse_visuelle.mp3",
        voice="verse",
    )
    return sortie_audio


if __name__ == "__main__":
    audio_path = analyse_visuelle_vers_audio(
        "img/dashboard_industrie.png",
        "Identifie les KPI principaux et donne les points de vigilance.",
    )
    print(f"Analyse audio disponible : {audio_path.resolve()}")
```

👉 **Exercice 6.1** — Ajoutez une étape de traduction (français → anglais) avant la synthèse vocale finalisée.

👉 **Exercice 6.2** — Industrialisez le pipeline via une CLI (`argparse`) ou une API FastAPI minimale.

---

## Pour aller plus loin

- Implémentez un chatbot vocal : streaming micro → transcription → réponse texte → synthèse vocale → restitution en temps réel.
- Comparez OpenAI avec un fournisseur alternatif (par exemple, `Deepgram` pour la transcription ou `Stability AI` pour l’image) et consignez vos observations.
- Mesurez et loggez les métriques d’usage (`usage.total_tokens`, temps de traitement, coûts estimés) pour préparer une mise en production.

👉 **Exercice final** — Concevez en binôme un mini-projet multimodal complet (ex. guide touristique audio illustré). Présentez aux autres groupes le pipeline, les modèles choisis, les limites identifiées et une démonstration.
