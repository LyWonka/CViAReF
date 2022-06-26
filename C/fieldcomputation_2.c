#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

// to be compiled using "gcc -shared -o summenfeld.so fieldcomputation_2.c"

// define struct for (x,y) tupel
struct xy {
	unsigned int x;
	unsigned int y;
};

struct E {
	float x;
	float y;
};

struct xy getXY(unsigned int index, unsigned int cols)
{
	struct xy pos = {index % cols, index / cols};
	return pos;
}

unsigned int getIndex(unsigned int x, unsigned int y, unsigned int cols)
{
	return (cols*y+x);
}

//float hypot(float x, float y) {
//    x *= x;
//    y *= y;
//    return(sqrt(x + y));
//}

struct E get_E(float q, float q_x, float q_y, float x, float y)
{
	if ( x == q_x && y == q_y){
		struct E e = {0.0,0.0};
		return(e);
	} else {
		float den = pow(hypot(x - q_x, y - q_y),3);
		struct E e = {q * (x - q_x) / den, q * (y - q_y) / den};
		return(e);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------
// Übergeben werden als array die Ladungsverteilung sowie zwei noch leere arrays, in denen dann die Feldkomponenten gespeichert werden
// -----------------------------------------------------------------------------------------------------------------------------------
int Summenfeld(float* charges, float* E_x, float* E_y, float* x_coords, float* y_coords, size_t rows, size_t cols, size_t x_coords_size, size_t y_coords_size)
{
	struct xy pos_charge;
	struct E e;
	for (int i = 0; i < (rows*cols); i++){
		// get charge
		float charge = charges[i];
		if (charge == 0){
			continue;
		}
		// get charge's position
		pos_charge = getXY(i, cols);
		// loop over the whole grid in order to update the electric field components
		for (int j = 0; j < x_coords_size; j++){
			float x = x_coords[j];
			for (int k = 0; k < y_coords_size; k++){
				float y = y_coords[k];
				unsigned int index = getIndex(k, j, cols);
				e = get_E(charge, x_coords[pos_charge.x], y_coords[pos_charge.y], x, y);
				E_x[index] += e.x;
				E_y[index] += e.y;
			}
		}
	}
}
