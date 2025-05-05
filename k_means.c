#include <stdio.h>
#include <stdlib.h>

void parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter);

int main(int argc, char *argv[])
{
   double **points = NULL;
   int n_points = 0; // Replace this later with actual point count
   int dim = 0;

   int K = 0;
   int max_iter = 0;

   int n_points = sizeof(points) / sizeof(points[0]);  

   // Parse command-line arguments
   parse_cmdline(argc, argv, n_points, &K, &max_iter);

   // Debug output to verify it worked
   printf("K = %d\n", K);
   printf("max_iter = %d\n", max_iter);

   return 0;
}

void parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter)
{
   if (argc != 2 && argc != 3)
   {
      fprintf(stderr, "Usage: ./kmeans K [1000]\n");
      exit(1);
   }

   *K = atoi(argv[1]);
   if (*K <= 1 || *K >= n_points)
   {
      fprintf(stderr, "Incorrect number of clusters!\n");
      exit(1);
   }

   if (argc == 3)
   {
      *max_iter = atoi(argv[2]);
      if (*max_iter <= 1 || *max_iter >= 1000)
      {
         fprintf(stderr, "Incorrect maximum iteration!\n");
         exit(1);
      }
   }
   else
   {
      *max_iter = 400; // default value
   }
}
