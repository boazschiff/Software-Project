#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>

double euclidean(const double *p1, const double *p2, int dim);
void kmeans(double **points, double **centroids, int n_points, int K, int dim, int max_iter, double eps);

static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *py_points, *py_centroids;
    int n_points, K, dim, max_iter;
    double eps = 1e-3;

    if (!PyArg_ParseTuple(args, "OOiii", &py_points, &py_centroids, &K, &max_iter, &dim))
        return NULL;

    n_points = PyList_Size(py_points);

    // Allocate C arrays
    double **points = (double **)malloc(n_points * sizeof(double *));
    double **centroids = (double **)malloc(K * sizeof(double *));

    for (int i = 0; i < n_points; i++) {
        PyObject *row = PyList_GetItem(py_points, i);
        points[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            points[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    for (int i = 0; i < K; i++) {
        PyObject *row = PyList_GetItem(py_centroids, i);
        centroids[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    // Run C kmeans
    kmeans(points, centroids, n_points, K, dim, max_iter, eps);

    // Create Python list to return
    PyObject *result = PyList_New(K);
    for (int i = 0; i < K; i++) {
        PyObject *row = PyList_New(dim);
        for (int j = 0; j < dim; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(centroids[i][j]));
        }
        PyList_SetItem(result, i, row);
    }

    // Free
    for (int i = 0; i < n_points; i++) free(points[i]);
    for (int i = 0; i < K; i++) free(centroids[i]);
    free(points);
    free(centroids);

    return result;
}

static PyMethodDef methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Run K-means clustering"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "kmeansmodule",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_kmeansmodule(void) {
    return PyModule_Create(&moduledef);
}
