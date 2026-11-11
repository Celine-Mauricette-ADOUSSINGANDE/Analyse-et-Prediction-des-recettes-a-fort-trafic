# 🍲 Analyse et prédiction des recettes à fort trafic

Projet de classification : prédire quelles recettes d'un site de cuisine généreront un **fort trafic** afin de choisir celles à mettre en avant pour favoriser la vente d'abonnements.

👉 **Application en ligne : [ouvrir l'application Streamlit](https://analyse-et-prediction-des-recettes-a-fort-trafic-xdpqeentzfoqe.streamlit.app/)**

## Contexte

Le site de recettes souhaite vendre des abonnements. Plus une recette attire de visites, plus la probabilité d'une vente augmente. L'objectif est donc de prédire la variable `high_traffic` à partir des caractéristiques de la recette.

## Données

Fichier : `recipe_site_traffic_2212.csv` (séparateur `;`), 947 recettes.

| Variable | Description |
|---|---|
| `recipe` | Identifiant unique de la recette |
| `calories` | Nombre de calories |
| `carbohydrate` | Glucides (g) |
| `sugar` | Sucre (g) |
| `protein` | Protéines (g) |
| `category` | Type de recette (10 catégories) |
| `servings` | Nombre de portions |
| `high_traffic` | « High » si le trafic était élevé (cible) |

## Méthodologie

1. **Nettoyage**
   - valeurs vides de `high_traffic` remplacées par « Low », puis codage 0 / 1 ;
   - fusion de `Chicken Breast` avec `Chicken` (11 → 10 catégories) ;
   - conversion de `servings` en entier (« 4 as a snack » → 4) ;
   - 52 recettes (5,5 %) sans données nutritionnelles : imputation par la médiane de la catégorie.
2. **Analyse exploratoire** : distributions, asymétrie (log1p), chi² et V de Cramér pour la catégorie, test de Mann-Whitney pour les variables nutritionnelles, étude des valeurs manquantes.
3. **Modélisation** : régression logistique, Random Forest et Gradient Boosting, dans des *pipelines* scikit-learn (prétraitement + modèle), validation croisée stratifiée à 5 plis, optimisation par GridSearch sur le F1, évaluation sur un jeu de test de 20 %.

## Principaux résultats

- **61 %** des recettes génèrent un fort trafic.
- La **catégorie** est la variable dominante (V de Cramér ≈ 0,58) : de 5 % (Beverages) à 99 % (Vegetable) de fort trafic.
- La nutrition et le nombre de portions apportent peu ; la variable `nutrition_missing` a été testée puis retirée (aucun gain en validation croisée).
- Résultats sur le jeu de test :

| Modèle | Accuracy | Précision | Rappel | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Régression logistique** | 0,779 | 0,848 | 0,774 | 0,809 | 0,869 |
| Random Forest | 0,753 | 0,788 | 0,809 | 0,798 | 0,852 |
| Gradient Boosting | 0,779 | 0,854 | 0,765 | 0,807 | 0,850 |

Les trois modèles sont équivalents ; la **régression logistique** est retenue pour sa simplicité et son interprétabilité.

## Structure du dépôt

```
.
├── app.py                       # Application Streamlit
├── recipe_site_traffic_2212.csv # Données
├── requirements.txt             # Dépendances Python
└── README.md
```

## Application Streamlit

L'application contient six pages : contexte, données et nettoyage, analyse exploratoire, modélisation, prédiction d'une nouvelle recette et conclusion. Chaque graphique est accompagné d'une interprétation.

### Lancer en local

```bash
git clone <url-du-depot>
cd <nom-du-depot>
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

### Déploiement

L'application est hébergée sur [Streamlit Community Cloud](https://share.streamlit.io). Après une période d'inactivité, elle peut se mettre en veille : le premier chargement peut alors prendre environ une minute.

## Limites et pistes d'amélioration

- Échantillon réduit (947 recettes, 190 en test) : les écarts entre modèles ne sont pas significatifs.
- Pas d'information sur les ingrédients ni sur le temps de préparation.
- L'imputation par médiane de catégorie est calculée avant la séparation entraînement / test (fuite minime) : à intégrer dans le pipeline.
- Le seuil de décision (0,5) peut être ajusté selon le coût d'une fausse promotion (précision) ou d'une bonne recette manquée (rappel).

## Technologies

Python · pandas · NumPy · SciPy · scikit-learn · matplotlib · seaborn · Streamlit
