#ifdef USE_CUDA

#include "math/Matrix.hpp"
#include "utils/Timer.hpp"
#include "nn/Initializers.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {
    Random gen(77);
    Matrix init1(32,32);
    Matrix init2(32,32);
    for (int i = 0; i < 16; i++) {
        Matrix init3 = Matrix::multiplyMatrixGPU(init1, init2);
    }
    std::vector<double> times_cpu;
    std::vector<double> times_gpu;
    std::vector<double> dim_matrix;

    double sec;
    Timer t;
    Index dim = 16;
    for (Index n = 0; n < 8; n++) {
        cout << "Calcul du temps de multiplication matricielle de 2 matrice carré de dimension " << dim << "X" << dim << endl;
        dim_matrix.push_back(dim);
        Matrix A(dim,dim);
        Matrix B(dim,dim);
        init_weight_He(A,gen);
        init_weight_He(B,gen);
        sec = 0;
        Index rep = (dim <= 256) ? 50 : (dim <= 1024 ? 20 : 10);

        Matrix::multiplyMatrixCPU(A, B);
        for (Index i = 0; i < rep; i++) {
            t.reset();
            Matrix::multiplyMatrixCPU(A,B);
            sec += t.elapsed();
        }
        times_cpu.push_back(sec/rep);

        sec = 0;
        Matrix::multiplyMatrixGPU(A, B);
        for (Index i = 0; i < rep; i++) {
            t.reset();
            Matrix::multiplyMatrixGPU(A,B);
            sec += t.elapsed();
        }

        times_gpu.push_back(sec/rep);
        dim *= 2;
    }

    ofstream log_file("data/benchmark_CPU_GPU.csv");
    if (log_file) {
        log_file << "Dimension,CPU,GPU\n";
    }
    else {
        cerr << "Erreur dans la création du fichier benchmark_CPU_GPU.csv" << endl;
    }

    for (Index i = 0; i < dim_matrix.size(); i++) {
        if (log_file) {
            log_file << dim_matrix[i] << "," << times_cpu[i] << "," << times_gpu[i] << "\n";
        }
    }
    return 0;
}

#else

#include <iostream>


int main() {
    using namespace std;
    cout << "CUDA non disponible" << endl;
    return 0;
}

#endif