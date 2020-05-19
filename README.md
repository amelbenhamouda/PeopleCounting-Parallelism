PeopleCounting-Parallelism
==========================
Projet M2.
L'objectif de ce projet est de se mettre dans le contexte d’une application simple afin d’appliquer par la suite la méthodologie d’optimisation des calculs / parallélisation à l’aide d’OpenMP.
L’application réalisée est un model de comptage de personnes dans un contexte simple, qui est développé en C++ à l’aide de la librairie OpenCV.

Utilisation
===========
## Compilation
	* g++ `pkg-config --cflags opencv` -fopenmp projet.cpp `pkg-config --libs opencv` -o projet

## Exécutable
	* ./projet

Notes
=====

## Authors
    * BEN HAMOUDA Amel (Université Paris-Est Marne-la-Vallee, France)
