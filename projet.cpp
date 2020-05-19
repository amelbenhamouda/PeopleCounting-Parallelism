/*
 * Fichier source pour le projet de comptage de personnes dans un contexte simple
 *---------------------------------------------------------------------------------------------------
 * Pour compiler :
 *     -> avec optimisation -O1 : g++ `pkg-config --cflags opencv` -fopenmp -O1  projet.cpp `pkg-config --libs opencv` -o projet
 *     -> avec optimisation -O2 : g++ `pkg-config --cflags opencv` -fopenmp -O2  projet.cpp `pkg-config --libs opencv` -o projet
 *     -> avec optimisation -O3 : g++ `pkg-config --cflags opencv` -fopenmp -O3  projet.cpp `pkg-config --libs opencv` -o projet
 *---------------------------------------------------------------------------------------------------
 * Auteur : Amel BEN HAMOUDA 03/2020
 */

/* 
 * Libraries stantards 
 */ 
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream> 
#include <vector> 

/* 
 * Libraries OpenCV "obligatoires" 
 */ 
#include "highgui.h"
#include "cv.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
  
/* 
 * Libraries time "obligatoires" pour le profiling
 */ 
#include <time.h>
#include <sys/time.h>

/* 
 * Libraries pour optimisation
 */ 
#include <sys/types.h>
#include <math.h>
#include <omp.h>
/* 
 * Définition des "namespace" pour évite cv::, std::, ou autre
 */  
using namespace std;
using namespace cv;
using std::cout;

/*
 *--------------- FUNCTIONS ---------------
 */
// Calcul de l’arrière plan qui s’actualise à chaque image
double soustractionFond(const vector<Mat> images, int N, int x, int y) {
	double first = 1.0 / N;
	double somme = 0.0;
	for (int i = 0; i < N; i++) {
		somme += images[i].at<uint8_t>(y, x); 
	}
	return first * somme;
}

// Calcul de la différence entre l’image d’arrière plan et l’image acquise
double detectionMouvement(const vector<Mat> images, int N, int x, int y, double soustFond) {
	double first = 1.0 / N;
	double somme = 0.0;
	for (int i = 0; i < N; i++) {
		somme += (images[i].at<uint8_t>(y, x) - soustFond) * (images[i].at<uint8_t>(y, x) - soustFond);
	}
	return sqrt(first * somme);
}

// Détection de mouvement significatif
Mat calculDetectionMouvement(const vector<Mat> images, int N, int rows, int cols, double seuil) {
	double soustFond;
	double detectMouv;
	Mat result = images[0];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			soustFond = soustractionFond(images, N, j, i);
			detectMouv = detectionMouvement(images, N, j, i, soustFond);
			
			if (detectMouv < seuil) {
				result.at<uint8_t>(i, j) = 0;
            }
            else {
                result.at<uint8_t>(i, j) = 255;
            }
		}
	}
	return result;
}

// Filtrage par ouverture 
Mat filtrageOuverture(Mat background) {
	Mat result = background;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE , Size(10, 10));//, Point(1,1));
    morphologyEx(background, result, MORPH_OPEN, kernel, Point(-1,-1), 2, BORDER_CONSTANT, morphologyDefaultBorderValue());
    // morphologyEx(background, result, MORPH_ERODE, kernel);
    // morphologyEx(background, result, MORPH_DILATE, kernel);
	return result;
}

// Comptage des composantes connexes restantes
int nbComposanteConnexe(Mat background) {
	vector<vector<Point>> contours;
    findContours(background, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
    int nb = contours.size();
    for (int i = 0; i < contours.size(); ++i) {
    	if (contours[i].size() < 40) {
    		nb --;
    	}
    }
    return nb;
}

/*
 *--------------- FUNCTIONS OPTIMISEE---------------
 */
// Calcul de la différence entre l’image d’arrière plan et l’image acquise
double detectionMouvementOpti(const vector<Mat> images, int N, int x, int y, double soustFond) {
	double first = 1.0 / N;
	double somme = 0.0;
	int calcul = 0.0;
	for (int i = 0; i < N; i++) {
		calcul = images[i].at<uint8_t>(y, x) - soustFond;
		somme += (calcul << calcul);
	}
	return sqrt(first * somme);
}

// Détection de mouvement significatif
Mat calculDetectionMouvementOpti(const vector<Mat> images, int N, int rows, int cols, double seuil) {
	double soustFond;
	double detectMouv;
	Mat result = images[0];
	#pragma omp collapse(3) private(i, j)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			soustFond = soustractionFond(images, N, j, i);
			detectMouv = detectionMouvementOpti(images, N, j, i, soustFond);

			if (detectMouv < seuil) {
				result.at<uint8_t>(i, j) = 0;
            }
            else {
                result.at<uint8_t>(i, j) = 255;
            }
		}
	}
	return result;
}

/*
 *--------------- MAIN FUNCTION ---------------
 */
int main () {
	VideoCapture videoCapture = VideoCapture(0); // le numéro 0 indique le point d'accès à la caméra 0 => /dev/video0
	if (!videoCapture.isOpened()) {
		cout << "Error opening video stream" << endl; 
		return -1;
	}

	Mat3b frame; // couleur
	Mat frame_gray; // niveau de gris 
	Mat background;
	vector<Mat> vectImage;
	
	unsigned char key = '0';
	int f = 0;
	double N = 10.0; // nombre d'images utilisé pour calculer l'arrière plan
	double valeurSigma = 3.0; // Petite valeur de sigma pour detection mouvement

	// Variables pour profiling
	struct timeval start, end;
	double e, s;
	int nbIter = 100;

	//----------------------------------------------------
	// Choix du rendu final
	std::cout << "Choisir le mode d'execution :" << std::endl;
	std::cout << "	1 pour un meilleur rendu, mais pas optimisé au niveau du temps d'execution," << std::endl;
	std::cout << "	2 pour un rendu moyen, mais optimisé au niveau du temps d'execution." << std::endl;
    int choice;
    std::cin >> choice;
    while (choice != 1 && choice !=2){
        std::cout << "Veuillez faire un choix : " << std::endl;
        std::cout << "	1 pour un meilleur rendu, mais pas optimisé au niveau du temps d'execution," << std::endl;
		std::cout << "	2 pour un rendu moyen, mais optimisé au niveau du temps d'execution." << std::endl;
        std::cin >> choice;
    }

	//----------------------------------------------------
	// Création des fenêtres pour affichage des résultats
	// vous pouvez ne pas les utiliser ou ajouter selon ces exemple
	cvNamedWindow("Video Input", WINDOW_NORMAL);
	cvNamedWindow("Video DetectionMvt", WINDOW_NORMAL);
	cvNamedWindow("Video FiltrageOuverture", WINDOW_NORMAL);
	// placement arbitraire des  fenêtre sur écran 
	// sinon les fenêtres sont superposée l'une sur l'autre
	cvMoveWindow("Video Input", 10, 30);
	cvMoveWindow("Video DetectionMvt", 400, 500);
	cvMoveWindow("Video FiltrageOuverture", 10, 500);
 
	// --------------------------------------------------
	// boucle infinie pour traiter la séquence vidéo  
	while (key != 'q' && key != 'Q') {

		// --------------------PROFILLING START----------------------------
	    	gettimeofday(&start, NULL);
	    // ------------------------------------------------
		// acquisition d'une trame video - librairie OpenCV
	    videoCapture.read(frame);
	    // ------------------------------------------------
			gettimeofday(&end, NULL);
		    e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
		    s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
		   // printf("Frame %d : videoCapture exec time : %lf us\n", f, (e - s));
		// --------------------PROFILLING END----------------------------

	    if (frame.empty()) {
	    	cout << "Frame is empty" << endl; 
	    	break;
	    }
	    //conversion en niveau de gris - librairie OpenCV
    	cvtColor(frame, frame_gray, CV_BGR2GRAY);

    	if (f < N) {
	    	// Ajout de N images dans un vecteur
		    vectImage.push_back(frame_gray);
		    f++;
		}
		else {
			// Supprime premier elem du vectImage
			vectImage.erase(vectImage.begin());
			// Ajout de la derniere acquisition dans vectImage
			vectImage.push_back(frame_gray.clone());


			// --------------------PROFILLING START----------------------------
		    	gettimeofday(&start, NULL);
		    // ------------------------------------------------
		    if (choice == 1) { // Calcul de la detection de mouvement
		    	background = calculDetectionMouvement(vectImage, N, frame_gray.rows, frame_gray.cols, valeurSigma);
		    	std::cout << "Frame " << f << " : calculDetectionMouvement exec time :";
		    }
		    else { // Calcul de la detection de mouvement Opti
		    	background = calculDetectionMouvementOpti(vectImage, N, frame_gray.rows, frame_gray.cols, valeurSigma);
		    	std::cout << "Frame " << f << " : calculDetectionMouvementOpti exec time :";
		    }
			// ------------------------------------------------
				gettimeofday(&end, NULL);
			    e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
			    s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
			    printf("%lf", e - s);
			// --------------------PROFILLING END----------------------------
			imshow("Video DetectionMvt", background);  


			// --------------------PROFILLING START----------------------------
		    	gettimeofday(&start, NULL);
		    // ------------------------------------------------
			// Filtrage par ouverture pour eliminer le bruit
			background = filtrageOuverture(background);
			// ------------------------------------------------
				gettimeofday(&end, NULL);
			    e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
			    s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
			    printf("Frame %d: filtrageOuverture exec time : %lf us\n", f, (e - s));
			// --------------------PROFILLING END----------------------------
			imshow("Video FiltrageOuverture", background); 


			// --------------------PROFILLING START----------------------------
		    	gettimeofday(&start, NULL);
		    // ------------------------------------------------
			// Comptage des composantes connexes
            cout << "Nombre de composante connexe : " << nbComposanteConnexe(background) << endl;
			// ------------------------------------------------
				gettimeofday(&end, NULL);
			    e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
			    s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
			    printf("Frame %d: nbComposanteConnexe exec time : %lf us\n", f, (e - s));
			// --------------------PROFILLING END----------------------------

			f++;
		}
		imshow("Video Input", frame);
		key = waitKey(5);
		cout << " " << endl;

		// nbIter--;
		// if (nbIter <= 0) {
		// 	break;
		// }
  	}
  	//Liberer l'objet capture video
  	videoCapture.release();
  	//Fermer toutes les fenetres
  	destroyAllWindows();
  	return 0;
}

