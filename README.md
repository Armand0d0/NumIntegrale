# NUMINTEGRALE
L'objecif était de trouver une intégrale dont la valeur serait un numéro de téléphone (ou toute autre suite de nombres prédéfinie) pour rendre ça possible, on représente un foction comme un arbre dont les noeuds peuvent avoir :
- 2 fils et représentent des foctions à deux paramètres  ( + -  * / )  
- 1 fils et représente une foction à un paramètre usuelle (cos, sin, tan, exp, ln, x -> x^a, ...)
- 0 fils et représent une constante ( des entiers, e, pi, ...) ou la variable x symbolisée par la fonction identité.

On utilise le bruteforce pour tester un maximum d'arbre et de bornes possible et pour chacune des fonctions on compare la différence entre les valeurs aux deux bornes, si le résultat est suffisemment proche de l'objectif, il suffit de dériver cette fonction pour avoir la fonction à intégrer entre ces bornes.

 On obtient de plutot bons résultats avec un peu de patience :


![alt text](./Int0.png?raw=true)


![alt text](./Int2.png?raw=true)

Saurez vous retrouver les primitives ?
