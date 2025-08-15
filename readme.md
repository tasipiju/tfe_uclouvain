
/////////////////////Réunion avec le prof: remarques & suggestions////////////////////////
1-Teams du 17/02/2025
    Recherche de sujet autour de "Création d'un tableau de bord":
        Suggestion 1 
            => App de gestion financière par les particuliers (comme Bilance) 
            => Déja existant (comme Monefy)
        Suggestion 2 
            => Traiter des indicateurs de macroéconomiques (eurostat, banque mondiale, etc.)
            => Traitement et prévision des marchés financiers
                => Série temporelles
                => Econométrie
            => Sources de données utiles (Fred, BCE, Bnaque mondiale, etc.)
    TAF:
        => Tester l'API de Fred
        => Chercher des sources de données intéressantes
        => Trouver un but pour chaque donnée à afficher dans le tableau de bord
        => Tester les technoloh-gies suivantes : FastAPI, Flask et Streamlit



2-Teams du 24/02/2025
    Présentation de ce que j'avais déja testé au prof
        Remarques du prof:
            - Corriger le problème de la non conversion de certaines (Couldn't convert string to float)
            - Pour le modèle ARIMA et pour tous les autres, mettre des champ permettant de tester chacun 
                en entrant ses propres paramètres (exple: p, d et q => afficher le graphique)
            - Donner la possibilité de choisir les dates de début et de fin qu'on veut afficher
            - Tester la correlation entre différentes courbes (s'inpirer de yardeni.com)
    TAF:
        => Voir comment amener le sujet, plusieurs suggestions comme:
            - Prédiction d'une série temporelle (exemple d'une variable macroéconomique)
            - Prédire l'évolution d'un marché financier comme le S&P 500 (variable macroéconomique)
            - Expliquer et afficher la correlation entre 2 séries temporelles
            - Prédire et/ou afficher des zones de recessions dans une série temporelle (zone grise)
            - Dévelloper et publier l'app pour le test
            - CONCLUSION => Analyser les points précédents et rédiger une 1ère idée de TFE
        => Lire un cours d'économétrie EDX (lien dans la conversation sur Teams): (EN COURS)
            - Précisement les chapitres 7 et 8 => 
            - Regarder aussi les variables macroéconomiques et le lien entre elles
        => Travail pratique:
            - Corriger l'erreur (Couldn't convert string to float)  (OK)
            - Ne pas afficher automatiquement les données dans l'interface graphique OK
                Solution => Mettre un bouton 
                    pour toutes les données, 
                    pour les 5 premières, 
                    pour les 5 dernières, 
                    pour les infos,
                    pour les statistiques.
            - Mettre la possibilité de choisir 
                L'intervalle de date à afficher (Date de début et de fin) OK
                La fréquence d'affichage (par mois / par an / par quartile / etc.) OK
            - Mettre la possibilité de :
                Documentation de FRED https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
                Afficher les recessions sur un graphique a coté du graphique OK
            - Mettre un menu déroulant pour les différents types de modèles
                Chaque modèle doit avoir des champs pour les paramètres permetant d'interagor avec l'application
                    Possibilité de modifier tous les paramètres sur le module python statmodels ()
                        ATTENTION Bien vérifier que les limites
                    Vérifier et afficher le cas ou il ya un modèle automatique (exple pour ARIMA, il ya pmdamira)
                    Solution => s'inspirer de:
                        1- VLAB
                            https://vlab.stern.nyu.edu/volatility/VOL.SVIX%3AUS-R.GARCH
                        2- L'application dévellopée par un autre étudiant pour le pricing des options dans l'UE
                            Tester l'application => http://derivativespricing.uclouvain.be/home
                            Mémoire => https://dial.uclouvain.be/memoire/ucl/object/thesis:29865
                Chaque model doit avoir un popup avec la documentation et les sources
                    Exemple sur le Mémoire => http://derivativespricing.uclouvain.be/home
                Chaque model doit présenter les différents tests possibles (S'inspirer de la vidéo yte de LeCoinStat)
                    Test Dickey-Fuller Test (vérification de la stationarité)
                        (si pas stationnaire => appliquer une différenciation et réafficher): tester avec 2 cas opposés
                    Test de Ljung-Box (Autocorrélation des résidus: le bruit blanc)
                    Test de Breusch-Pagan (vérifier l'homoscédasticité)
                    Test de Shapiro-Wilk (Normalité)
                        Afficher aussi un histogramme ou un Q-Q plot des résidus
                    ACF
                    PACF
                    MSE
                    RMSE
                    MAE
                    R carré
                    Test de Jarque-Bera (Normalité des résidus)
                    Test d’hétéroscédasticité (Breusch-Pagan, ARCH)
                Chaque modèle doit faire les prévisions
                    Possibilité de choisir la taille de l'entrainement (entre 60 et 80)et de test (le rester)
                        ATTENTION véfifier que la totalité donne 100%
                        entrainement du modèle + prévision sur 
                    Possiblité de choisir la taille des données à prédire (dépend du modèle)
                    Tester le modèle à travers
                        Matrice de confusion (VP,VN,FP,FN)
                        Bon article sur le lien entre statistique et Machine learning
                            https://www.clicdata.com/fr/blog/role-statistiques-machine-learning/
            - Mettre plusieurs graphiques sur un meme shéma (en fonction des cas): OPTIONS AVANCÉSs
                Exemple sur => https://yardeni.com/our-charts/



2-Teams du 02/05/2025
    Présentation de ce que j'avais déja testé au prof (Uniquement THEORIQUE)
        Voici ses remarques et suggestions qui guideront la suite du travail
            -Se focaliser sur ce que je vais faire
            -Ne pas traiter les éléments de facon isolée (erreur => toute la partie macro)
            -Tous ce qui est traiter théoriquement (empirique) doit se retrouver dans le TDB
            -Se limter à 2 ou 3 points car il ya beaucoup de travail
                --machine learning (AR, MA, ARIMA, etc....)
                --modèles économétriques
            -Utiliser des études/articles scientifiques 
            -ATTENTION:
                Delimiter le scope
                Les choses sont présentés comme si on lisait un slide de macroéconomie
                Toujours ajouter des références des articles utilisés
    TAF:
        -Dans le tableau de bord (TDB): 
            Modifer l'affichage
                S'inspirer du TFE d'un ancien étudiant (pour univarié)
                S'inspirer de Yardeni.com (pour bivarié)
            Traiter les 03 onglets:
                Data:
                    plus rien à faire ici OKKKKKK
                Analyse univariée (prédiction des séries temporelles par rapport à elle meme)
                    modèles de machine learning (AR, MA, ARMA, ARIMA)                   
                    modèles de deep learning (modèle LSTM) 
                Analyse bivariée
                    Affichage (scatter plot)
                    modèles de machine learning (VAR, VECM, ARDL)
        -Faire des analyses
            A partir de ce qu'on a dit sur le plan théorique, tester des théories simples de macroéconomie
                (par exemple: voici ce que dit la théorie et voici ce qu'on obtien en pratique)
            Toujours lié la théorie à la pratique
        -DÉCISION POUR LA SUITE (théorie et pratique):
            -Nouvelle vesion word (beaucoup de modifications (onglets, contenus et corrections))
            -Partie univariée (inflation)
                Trouver des articles scientifiques qui l'explique mieux (avec des statistiques)
                Montrer à travers le TDB comment prédire l'inflation par rapport à elle meme (ML)
            -Partie bivavariée
                Choix de grandeurs:
                    macro-micro => inflation et taux d'interet
                    macro-macro => inflation et taux de chomage
                Afficher des 2 graphiques sur la meme courbe pour voir l'évolution (scatter plot)
                Developper dans le TDB
                Tester et expliquer les résultats des modèles issus du TDB



3-Teams du 01/07/2025
    Voici les critiques et suggestions du prof sur le travail réalise et la suite (par partie)
        -Toute la partie texte:
            --Revoir la numérotation (Maximum 3 niveaux de niveaux)
            --passer au format APA: c'est le format date-auteur (ne pas utilsier des numéros)
            --Relire pour les fautes d'orthographe
            --Utiliser la page de garde de la LSM
            --Limiter les résulats chiffrés à 2 décimales (au lieu de 14)
        -Partie Introducion
            --Repositionner le mémoire dans un contexte open data (l'idée de faire un mémoire c'est aussi de profiter des données open source)
            --Trouver un angle de positionnement de la partie application (a qui est déstiné l'app??)
                Professeurs / Etudiant / Economistes etc...
        -Partie théorique:
            --Comment aborder la théorie tout en restant focaliser sur ce qui sera fait dans l'app
                Exemple: Réduire voire supprimer ou mettre en annexe tout ce qui n'est pas utilisé dans l'app
                PROBLEME: Le mémoire commence réellement dans la partie "prédiction de l'inflation"
                SOLUTION: comment positionner le document??
                    On a à la fois les concepts macro-économiques et économétriques
                    Est-ce que on va plutot tendre vers une vulgarisation des 
                        Exemple => vulgarisation des concepts écononométriques (appliqué à la macroéconomie)
                        Exemple => introduction à la macroéconomie grace à l'économétrie
                        etc.... (a réfléchir)
            --Réflechir (choix) si on fait un dashboard Européen/Belge/Mondial???
                (Faire un choix et rester cohérent tout au long du développement)
                Trouver un autre site pour importer des données
                    PROBLEME: FredAPI est fortement US => Eurostat peur etre une bonne solution
            --Trouver des articles plus techniques pour la partie théorie
                (aller sur google scholar ou ce que le prof m'a envoyé)
                1-trouver 5 articles scientifiques macro-macro (lien entre inflation et taux de d'interet)
                2-trouver 5 articles scientifiques macro-micro (lien entre inflation et taux de chomage)
                    Exemple: courbe de Phillips => Comment les articles scientifoques présente la chose
        -Partie pratique:
            --Limiter les résultats à 2 ou 4 décimales (au lieu de 14)
            --PAS trop se focaliser sur comment on a procédé (se focaliser sur les résultats)
                Exemple: comment on a fait pour avoir la clé API etc...
                Bref ne pas trop mettre en avant les considérations terre à terre => Rester high level
            --Présentation des résultats:
                Mettre uniquement les graphiques générées (PAS DE PRINTSCREEN DE L'APP)
                    Inclure uniquement les graphiques
                    Faire un tableau de comparaison comme sur des articles scientifiques
                Consacrer un chapitre entier sur la partie application qui contiendra:
                    A qui l'application est dédié??
                    Le choix techniques effectués
                    Comment accéder à l'application et comment l'utiliser?
                    Comment l'utiliser?
    Informations utiles
        Pas de présentation orale pour le master 60 
            Donc il faut tout mettre dans une bonne rédaction du document à déposer
        2 enseignants doivent évaluer le document (rendre le document très clair)
    STRUCTURATION DE L'INFORMATION GLOBALE (MEMOIRE):
        1-Introduction
        2-Revue de littérature
            -02 parties grandes parties:
                prévision univariée
                prévision bivariée
            présenter un résumé des articles scientifiques
            pour cloturer cette partie: ce qui m'interesse c'est l'inflation de manière univariée, puis de facon bivariée(macro-macro/macro-micro)
        3-Partie empirique
            -02 parties grandes parties:
                prévision univariée
                prévision bivariée
            Pour chaque modèle, faire un flowchart du fonctionnement du modèle (théorie/pratique)
            présenter les modèles (prédiction des séries temporelles)
            présenter les résulats des différents modèles
            Faire une comparaison (tableau avec RMSE) => un peu comme le fait les articles scientifiques
        4-Implémentation via le dashboard (permet de présenter les résultats)
            -Introduction
                Importance => montrer en quoi c'est utile à un prof de macroéconomie ou d'économétrie???
            -Choix techniques
            -Fonctionnement du dashboard
                les différents onglets et ce qu'ils font
            -Comment l'utiliser?
            -Voici les cas pratiques (en annexes peut etre)
        5-Conclusion

            




/////////////////////////////////////Tools/////////////////////////////////////////////
1-Fred API
    Catégories de l'API
        https://fred.stlouisfed.org/categories
    API
        https://fred.stlouisfed.org/docs/api/fred/

2-Streamlit: créer une interface graphique pour publier sonnmodèle de Machine learning
    -Comprendre streamlit (les bases)
        https://www.youtube.com/watch?v=7pbhs1jLhRA
    -Petite app avec plusiuers pages
        https://www.youtube.com/watch?v=bybdfk3UVT0
    -exemple de projet (Réalisation d'une application de Machine Learning avec Streamlit et Scikit-learn)
        Partie 1: écriture du code
            https://www.youtube.com/watch?v=WWYdwyhDgmY
        Partie 2: déploiement de l'application
            https://www.youtube.com/watch?v=wjRlWuXmlvw

3-Fast API: créer des api pour le Machine learning
    https://www.youtube.com/watch?v=0-yncL0bqZs

4-Flask:
    -Comprendre les bases
        en fancais => https://www.youtube.com/watch?v=Ihp_cG7c2Rk&list=PLV1TsfPiCx8PXHsHeJKvSSC8zfi4Kvcfs&index=1
        en anglais => https://www.youtube.com/watch?v=dam0GPOAvVI
        cours de freeCodeCamp => https://www.youtube.com/watch?v=Qr4QMBUPxWo&t=8737s
    -Projet 1
        -Code
            https://www.youtube.com/watch?v=mZKOEvbGGwQ&t=4106s
        -Déploiement
            https://www.youtube.com/watch?v=mNATOPXKqj8
    -Projet 2: avec TaiwlindCSS & ChatGPT: clone de ChatGPT
        https://www.youtube.com/watch?app=desktop&v=AYmcV3b7lWQ&t=10084s





//////////////////Tutoriels ou documentation utiles pour le mémoire ou astuces/////////////////////
1-Théorie sur les séries Séries Temporelles
    Vidéo 1 - explications sur les séries temporelles de Datascientest (tp avec le modèle ARIMA)
        https://www.youtube.com/watch?v=z20vB4lBNmg
    Vidéo 2 - explications brèves sur les séries temporelles de Datascientest (tp avec le modèle LSTM)
        https://www.youtube.com/watch?v=R8ZwPRmt9Bc
    Comprendre les notions de tendance, saisonnalité, bruit, modèle exponentiel et modèle ARIMA
        https://blog.statoscop.fr/timeseries-4.html
    Les différents types de modèles des séries temporelles: 
        -classiques ou statistiques(moyenne mobile, lissage exponentiel, ARIMA, SARIMA, TBATS);
        -apprentissage automatique(Régression linéaire, XGBoost, Random Forest);
        -apprentissage profond (RNN, LSTM).
        https://www.datacamp.com/fr/tutorial/tutorial-time-series-forecasting?dc_referrer=https%3A%2F%2Fwww.google.com%2F
    Quelques explications sur l'ACF, PACF, Test de Ljung-Box, homoscédacité, stationarité, Différentciation etc... 
        Vidéo youtube également
        Méthodologie de Box-Jenkins le modèle ARIMA
            https://github.com/LeCoinStat/100JoursDeML/blob/main/07_Series_Temporelles/01_ARIMA/01_Modele_Arima.ipynb

2-Autres concepts utiles au mémoire
    Statistiques
        - Les bases
            https://www150.statcan.gc.ca/n1/edu/power-pouvoir/toc-tdm/5214718-fra.htm
        - Les tests statistiques (pratique avec R)
            https://www.sthda.com/french/wiki/logiciel-r
        - Grosse documentation sur les statistiques (statistiques descriptive, analyse statistique)
            https://fr.statisticseasily.com/glossaire/
        - Lien entre statistique et machine learning
            https://www.clicdata.com/fr/blog/role-statistiques-machine-learning/
    Exercice: Comment prédire le prix d'une action en bourse (séries temporelles)
        https://www.youtube.com/watch?v=ACi3aiJRIP8
    Streamlit
        -Comprendre streamlit (les bases)
            https://www.youtube.com/watch?v=7pbhs1jLhRA
        -Petite app avec plusiuers pages
            https://www.youtube.com/watch?v=bybdfk3UVT0
        -exemple de projet (Réalisation d'une application de Machine Learning avec Streamlit et Scikit-learn)
            Partie 1: écriture du code
                https://www.youtube.com/watch?v=WWYdwyhDgmY
            Partie 2: déploiement de l'application
                https://www.youtube.com/watch?v=wjRlWuXmlvw
    Fast API: créer des api pour le Machine learning
        https://www.youtube.com/watch?v=0-yncL0bqZs
    Flask-TaiwlindCSS-ChatGPT-Python: clone de ChatGPT
        https://www.youtube.com/watch?v=AYmcV3b7lWQ

2-Comment importer une fonction d'un fichier python d'un dossier dans un autre fichier d'un autre dossier
    https://www.geeksforgeeks.org/python-import-module-from-different-directory/

3-Inspiration pour le code de base pour:
    -fredapi
        https://github.com/AdamGetbags/fredAPI/blob/main/fred_api.py
    -modèle arima
        https://github.com/LeCoinStat/100JoursDeML/blob/main/07_Series_Temporelles/01_ARIMA/01_Modele_Arima.ipynb

4-ATTENTION parfois pandas supprime les index, du coup plotly n'ARRIVE PAS à les afficher
    SOL: 
        1-afficher les colonnes
        2-vérifier que certaines colonnes ne sont pas des index
            SI oui, supprimer l'idexation via df = df.reset_index()



