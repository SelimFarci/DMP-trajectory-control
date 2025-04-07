# DMP-trajectory-control
L’objectif du TP est de contrôler le robot Niryo avec l’algorithme DMP (Ijspeert, Schaal, Atkeson). L’algorithme DMP fait un contrôle de la trajectoire comme un système dynamique. DMP utilise les équations de système linéaire de premier ordre et de second ordre pour simuler un système physique.

l’algorithme DMP se sert de ces équations pour contrôler ou encoder le comportement nonlinéaire
du robot. DMP ajoute de la flexibilité au robot qui peut se permettre des erreurs.
Les trajectoires sont approximées par des combinaisons linéaires de fonctions gaussiennes.


Le but du projet est de concevoir un contrôleur DMP : 
1) implémenter un système dynamique et faites un contrôle du robot ou en simulation en
utilisant le modèle de cinématique directe et inverse du robot pour suivre la trajectoire.
Spécifier alpha, beta. On n’approximera pas avec des fonctions gaussiennes. Faites
démarrer de plusieurs points initiaux différents et sauvegarder les trajectoires.
2) Cette fois-ci, approximer une trajectoire quelconque ou que vous aurez spécifié avec des
fonctions gaussiennes et sans poids W (on ne fera pas de la loi d’optimisation).
3) Faire un cycle limite (trajectoire rythmique).
