#include <stdio.h>

int main() {
   double **points = NULL;
   int n_points;
   int dim =0;
   int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr);

   // Read all points from stdin
   if (read_points(&points, &n_points, &dim) != 0) {
    return 1;  // error already printed
}




   return 0;
}
