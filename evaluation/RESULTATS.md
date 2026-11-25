# Résultats d'évaluation — MeetMind

## 1. Données

| Jeu | Contenu | Usage |
|---|---|---|
| `data/corpus/train/` | 10 réunions synthétiques, 146 phrases étiquetées | Entraînement |
| `data/annotations/seed.jsonl` + `reunions_exemple.csv` | 285 phrases écrites et vérifiées à la main | Entraînement |
| `data/corpus/test/` | **5 réunions synthétiques, 70 phrases**, avec résumé, décisions et actions de référence | **Test uniquement** |

**Important :** toutes les réunions sont **synthétiques**, rédigées pour le projet (domaines variés :
cabinet comptable, université, e-commerce, marketing en darija, association). Les 5 réunions de test
n'ont été utilisées **ni pour l'entraînement ni pour écrire les règles**. Elles utilisent volontairement
des formulations différentes pour mesurer la généralisation.

Date de référence des réunions : lundi 16/11/2026 (pour résoudre « vendredi », « demain »…).

## 2. Résultats sur le jeu de test (5 réunions)

| Configuration | ROUGE-L résumé | F1 décisions | F1 actions | Responsable correct | Échéance correcte |
|---|---|---|---|---|---|
| Règles seules + TextRank | **0,357** ✅ | 0,53 | 0,32 ❌ | 100 % | 100 % |
| TF-IDF + régression logistique, règles si incertain (seuil 0,6) — **config. par défaut** | **0,357** ✅ | **0,65** | 0,49 ❌ | 100 % | ~90 % |
| TF-IDF seul, sans règles *(analyse a posteriori)* | **0,357** ✅ | 0,46 | **0,81** ✅ | — | — |
| **CamemBERT fine-tuné + BARThez** — **configuration finale** | 0,247 ❌ | **0,76** | **0,80** ✅ | 100 % | ~93 % |

Résumé seul : TextRank (extractif) ROUGE-L **0,357** ✅ · BARThez (abstractif) **0,247**
(par réunion : 0,32 · 0,34 · 0,32 · **0,06 (darija)** · 0,19).

Objectifs du cahier des charges : ROUGE-L ≥ 0,35 · F1 actions ≥ 0,75.

Classification des phrases (70 phrases de test) :

| Classifieur | F1 décision | F1 action |
|---|---|---|
| Règles | 0,59 | 0,50 |
| TF-IDF + règles si incertain | 0,74 | 0,73 |
| TF-IDF seul | 0,53 | 0,90 |
| **CamemBERT fine-tuné** (Colab, GPU T4, 8 époques) | **0,82** | **0,90** |

Fichiers : `resultats_corpus_test_regles.json`, `resultats_corpus_test_tfidf.json`, `analyse_corpus_test_tfidf_seul.json`.

Exemple qualitatif (réunion de test `test_03`) : avec CamemBERT, l'application retrouve **2/2 décisions
et 3/3 actions** avec le bon responsable et la bonne échéance (TF-IDF : 1/2 et 1/3).

## 3. Analyse

1. **Le résumé atteint l'objectif** (ROUGE-L 0,357) même avec un résumé extractif (TextRank).
   BARThez / mT5 restent à mesurer sur GPU (notebook Colab, section 3).
2. **Les règles généralisent mal** : très précises sur les réunions qui ont servi à les écrire
   (F1 actions 1,0 sur `data/samples/`), elles tombent à **0,32** sur des réunions nouvelles.
   C'est l'argument principal pour l'apprentissage automatique.
3. **Un simple modèle appris (TF-IDF) fait passer le F1 actions de 0,32 à 0,81**, avec seulement
   ~430 phrases d'entraînement et quelques secondes sur CPU.
4. **Compromis décisions / actions** : la combinaison « modèle + règles quand le modèle hésite »
   améliore les décisions (0,65) mais dégrade les actions (0,49). Le seuil a été choisi par validation
   croisée sur les données d'entraînement (où il était le meilleur) ; le test montre qu'il ne
   généralise pas. On ne l'a pas modifié après coup pour ne pas « régler sur le test ».
   Le seuil est configurable : `MODEL_MIN_CONFIDENCE=0` → modèle seul.
5. **Responsables et échéances** : quand l'action est trouvée, le responsable est correct dans 100 %
   des cas et l'échéance dans ~90 %.
6. **Résumé abstractif (BARThez) vs extractif (TextRank)** : BARThez rédige avec ses propres mots,
   ce qui est plus lisible mais pénalisé par ROUGE (qui compte les mots communs avec la référence).
   BARThez a été entraîné sur des articles de presse (OrangeSum), pas sur des réunions, et ne comprend
   pas la darija (ROUGE-L 0,06 sur la réunion en darija). Il peut aussi « halluciner »
   (ex. « les employés de MaroColis »). Pour atteindre l'objectif ROUGE-L ≥ 0,35 : `SUMMARIZER=extractive`.
7. **CamemBERT est le meilleur classifieur** : F1 décisions 0,76 et actions 0,80 au niveau des réunions
   (0,82 / 0,90 au niveau des phrases), contre 0,53 / 0,32 pour les règles.
8. **Limites** : corpus petit et synthétique ; transcriptions sans erreurs de reconnaissance ;
   temps de traitement d'une heure d'audio non mesuré sur GPU.

## 4. Prochaines étapes

- ~~Fine-tuner CamemBERT~~ ✅ fait (`notebooks/MeetMind_CamemBERT_autonome.ipynb`) : meilleur modèle.
- Ajouter des **réunions réelles** annotées.
- Mesurer BARThez / mT5 et le temps de Whisper sur GPU.

## 5. Reproduire

```bash
python data/corpus/build_corpus.py
python training/train_tfidf.py --data data/corpus/annotations_train.csv data/annotations/seed.jsonl data/annotations/reunions_exemple.csv --eval data/corpus/annotations_test.csv
NLP_MODE=light python evaluation/evaluate.py data/corpus/test/*.gold.json --sentences data/corpus/annotations_test.csv
NLP_MODE=auto  python evaluation/evaluate.py data/corpus/test/*.gold.json --sentences data/corpus/annotations_test.csv
```
