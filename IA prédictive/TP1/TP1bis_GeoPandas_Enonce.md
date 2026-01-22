# TP1bis — Introduction à GeoPandas (≈ 2h30)

[⬅️ Retour au sommaire](../../LISEZMOI.md)

## Objectifs

- Comprendre les données géospatiales vectorielles (points, lignes, polygones)
- Manipuler des GeoDataFrames avec GeoPandas
- Lire et écrire des fichiers géographiques (GeoJSON, Shapefile, GeoPackage)
- Effectuer des opérations spatiales (intersections, buffers, jointures)
- Produire des cartes thématiques avec matplotlib

## Prérequis

- Python 3.x, geopandas, matplotlib, shapely
- Avoir suivi le TP1 sur pandas

```bash
pip install geopandas matplotlib shapely
```

---

## Partie 1 — Premiers pas avec GeoPandas (20 min)

1. Charger le jeu de données intégré `naturalearth_lowres` (pays du monde) :

   ```python
   import geopandas as gpd
   world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
   ```

2. Afficher les 5 premières lignes et explorer les colonnes
3. Identifier le type de géométrie (`geometry.type`)
4. Afficher `world.crs` (système de coordonnées)
5. Tracer une carte simple avec `world.plot()`

---

## Partie 2 — Exploration et sélections (20 min)

1. Filtrer les pays d'Europe (`continent == "Europe"`)
2. Filtrer les pays avec une population > 50 millions
3. Sélectionner la France et afficher sa géométrie
4. Calculer l'aire de chaque pays (`geometry.area`) — noter les unités !
5. Trier les pays par superficie décroissante

---

## Partie 3 — Systèmes de coordonnées (CRS) (20 min)

1. Vérifier le CRS actuel du GeoDataFrame
2. Reprojeter en Lambert-93 (EPSG:2154) pour la France
3. Reprojeter en projection équivalente (EPSG:6933) pour calculer les aires en m²
4. Comparer les aires avant/après reprojection
5. Comprendre la différence entre `to_crs()` et `set_crs()`

---

## Partie 4 — Création de géométries (25 min)

1. Créer un GeoDataFrame avec 5 villes françaises (points) :

   | Ville | Longitude | Latitude |
   |-------|-----------|----------|
   | Paris | 2.3522 | 48.8566 |
   | Lyon | 4.8357 | 45.7640 |
   | Marseille | 5.3698 | 43.2965 |
   | Toulouse | 1.4442 | 43.6047 |
   | Bordeaux | -0.5792 | 44.8378 |

2. Définir le CRS en WGS84 (EPSG:4326)
3. Tracer les villes sur une carte de France
4. Créer des buffers de 50 km autour de chaque ville
5. Visualiser les zones tampons

---

## Partie 5 — Opérations spatiales (25 min)

1. **Intersection** : trouver les pays traversés par une ligne (équateur simplifié)
2. **Contains/Within** : vérifier si un point est dans un polygone
3. **Distance** : calculer la distance entre Paris et les autres villes
4. **Centroïde** : calculer le centroïde de chaque pays européen
5. **Union** : fusionner tous les pays de l'UE en un seul polygone

---

## Partie 6 — Jointures spatiales (20 min)

1. Charger le jeu de données des villes `naturalearth_cities` :

   ```python
   cities = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))
   ```

2. Effectuer une jointure spatiale pour associer chaque ville à son pays
3. Compter le nombre de villes par pays
4. Identifier les pays sans ville dans le dataset

---

## Partie 7 — Lecture/écriture de fichiers (15 min)

1. Exporter les pays européens en GeoJSON :

   ```python
   europe.to_file("europe.geojson", driver="GeoJSON")
   ```

2. Exporter en Shapefile (noter les fichiers créés)
3. Exporter en GeoPackage (format recommandé)
4. Recharger et vérifier l'intégrité des données

---

## Partie 8 — Cartographie thématique (25 min)

### 8.1 Carte choroplèthe

- Colorier les pays selon leur population (`column="pop_est"`)
- Ajouter une légende avec `legend=True`

### 8.2 Carte avec catégories

- Colorier par continent avec des couleurs distinctes

### 8.3 Carte multi-couches

- Superposer pays + villes + buffers sur une même figure

### 8.4 Personnalisation

- Ajouter un titre, modifier les couleurs de bordure
- Utiliser `figsize`, `edgecolor`, `linewidth`
- Ajouter des annotations pour les capitales

### 8.5 (Bonus) Export haute résolution

- Sauvegarder la carte en PNG 300 DPI

---

## Partie 9 — Exercices bonus (optionnel)

1. **Analyse de densité** : calculer la densité de population par pays et créer une carte choroplèthe
2. **Plus proche voisin** : pour chaque ville, trouver la ville la plus proche
3. **Découpage** : extraire uniquement les parties des pays situées dans une bounding box
4. **Agrégation** : regrouper les pays par continent et calculer la population totale

---

## Conseils

- Toujours vérifier le CRS avant les opérations spatiales
- Pour les calculs de distance/aire précis, reprojeter dans un CRS métrique
- `explore()` (si folium installé) permet une visualisation interactive
- La colonne `geometry` est spéciale : c'est elle qui fait la différence avec pandas

## Ressources

- [Documentation GeoPandas](https://geopandas.org/)
- [Shapely User Manual](https://shapely.readthedocs.io/)
- [EPSG.io](https://epsg.io/) pour les systèmes de coordonnées

## Durée estimée

- 2h30 (parties 1-8)
- +30 min pour les bonus

---

📘 **[Accéder au corrigé](TP1bis_GeoPandas_Corrige.ipynb)**
