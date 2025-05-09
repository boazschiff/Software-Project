#include <stdio.h>
#include <stdlib.h>

void parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter);
int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr);
#define MAX_LINE_LEN 1024

int main(int argc, char *argv[])
{
    double **points = NULL;
    int n_points = 0; // Replace this later with actual point count
    int dim = 0;

    if (read_points(&points, &n_points, &dim) != 0)
    {
        return 1; // error already printed
    }

    int K = 0;
    int max_iter = 0;

    int n_points = sizeof(points) / sizeof(points[0]);

    // Parse command-line arguments
    parse_cmdline(argc, argv, n_points, &K, &max_iter);

    // Debug output to verify it worked
    printf("K = %d\n", K);
    printf("max_iter = %d\n", max_iter);
}

int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr)
{
    char line[MAX_LINE_LEN];
    int n_points = 0;
    int capacity = 10;
    int dim = 0; /* local working copy of dimension */

    double **points = (double **)malloc(capacity * sizeof(double *));
    if (points == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    while (fgets(line, MAX_LINE_LEN, stdin) != NULL)
    {
        int i, current_dim = 0;
        char *token;
        char *line_copy;
        char *tmp;

        line[strcspn(line, "\r\n")] = '\0';
        line_copy = (char *)malloc(strlen(line) + 1);
        if (line_copy == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            return 1;
        }
        strcpy(line_copy, line);

        token = strtok(line, ",");
        while (token != NULL)
        {
            current_dim++;
            token = strtok(NULL, ",");
        }

        if (n_points == 0)
        {
            dim = current_dim;
        }
        else if (dim != current_dim)
        {
            fprintf(stderr, "Inconsistent dimensions at line %d.\n", n_points + 1);
            free(line_copy);
            return 1;
        }
        points[n_points] = (double *)malloc(dim * sizeof(double));
        if (points[n_points] == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            free(line_copy);
            return 1;
        }
        token = strtok(line_copy, ",");
        for (i = 0; i < dim && token != NULL; i++)
        {
            points[n_points][i] = strtod(token, &tmp);
            token = strtok(NULL, ",");
        }

        if (i != dim)
        {
            fprintf(stderr, "Line %d: expected %d values but got %d.\n", n_points + 1, dim, i);
            free(line_copy);
            return 1;
        }

        free(line_copy);
        n_points++;
        /* Resize points array if needed */
        if (n_points >= capacity)
        {
            double **new_points;
            capacity *= 2;
            new_points = (double **)realloc(points, capacity * sizeof(double *));
            if (new_points == NULL)
            {
                fprintf(stderr, "Reallocation failed.\n");
                return 1;
            }
            points = new_points;
        }
    }

    if (n_points == 0)
    {
        fprintf(stderr, "No input provided.\n");
        return 1;
    }
    *points_ptr = points;
    *n_points_ptr = n_points;
    *dim_ptr = dim;

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
