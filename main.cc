#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

// Function declarations
void PrintMatrix(const float *matrix, int dim);
void PrintResult(const std::vector<float> &matrix, int dim, double elapsed_time,
                 int process_id, bool print_original, bool print_answer);

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  const int kDim = 2000;
  const int kRowsPerProcess = kDim / num_processes;

  int process_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
  const int kStartRow = process_id * kRowsPerProcess;
  const int kEndRow = kStartRow + kRowsPerProcess;

  std::vector<float> matrix;
  std::vector<float> matrix_chunk(kDim * kRowsPerProcess);
  std::vector<float> pivot_row(kDim);

  if (process_id == 0) {
    std::cout << "Dimension: " << kDim << std::endl;
    std::mt19937 mt(1000);
    std::uniform_real_distribution<float> dist(-10.0f, 200.0f);

    matrix.resize(kDim * kDim);
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });
  }

  MPI_Scatter(matrix.data(), kDim * kRowsPerProcess, MPI_FLOAT,
              matrix_chunk.data(), kDim * kRowsPerProcess, MPI_FLOAT, 0,
              MPI_COMM_WORLD);

  std::vector<MPI_Request> requests(num_processes);
  double start_time = MPI_Wtime();

  for (int row = 0; row < kEndRow; ++row) {
    int mapped_rank = row / kRowsPerProcess;

    if (process_id == mapped_rank) {
      int local_row = row % kRowsPerProcess;
      float pivot = matrix_chunk[local_row * kDim + row];

      for (int col = row; col < kDim; ++col) {
        matrix_chunk[local_row * kDim + col] /= pivot;
      }

      for (int i = mapped_rank + 1; i < num_processes; ++i) {
        MPI_Isend(matrix_chunk.data() + kDim * local_row, kDim, MPI_FLOAT, i, 0,
                  MPI_COMM_WORLD, &requests[i]);
      }

      for (int elim_row = local_row + 1; elim_row < kRowsPerProcess;
           ++elim_row) {
        float scale = matrix_chunk[elim_row * kDim + row];
        for (int col = row; col < kDim; ++col) {
          matrix_chunk[elim_row * kDim + col] -=
              matrix_chunk[local_row * kDim + col] * scale;
        }
      }

      for (int i = mapped_rank + 1; i < num_processes; ++i) {
        MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
      }
    } else {
      MPI_Recv(pivot_row.data(), kDim, MPI_FLOAT, mapped_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int elim_row = 0; elim_row < kRowsPerProcess; ++elim_row) {
        float scale = matrix_chunk[elim_row * kDim + row];
        for (int col = row; col < kDim; ++col) {
          matrix_chunk[elim_row * kDim + col] -= pivot_row[col] * scale;
        }
      }
    }
  }

  double end_time = MPI_Wtime();
  double elapsed_time = end_time - start_time;

  MPI_Gather(matrix_chunk.data(), kRowsPerProcess * kDim, MPI_FLOAT,
             matrix.data(), kRowsPerProcess * kDim, MPI_FLOAT, 0,
             MPI_COMM_WORLD);

  bool print_original = false;
  bool print_answer = false;
  if (process_id == 0) {
    char response;
    std::cout << "Print the original matrix? (Y/n): ";
    std::cin >> response;
    print_original = (response == 'Y' || response == 'y');

    std::cout << "Print the result matrix? (Y/n): ";
    std::cin >> response;
    print_answer = (response == 'Y' || response == 'y');
  }

  PrintResult(matrix, kDim, elapsed_time, process_id, print_original,
              print_answer);

  MPI_Finalize();
  return 0;
}

void PrintMatrix(const float *matrix, int dim) {
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      std::cout << matrix[i * dim + j] << ' ';
    }
    std::cout << '\n';
  }
}

void PrintResult(const std::vector<float> &matrix, int dim, double elapsed_time,
                 int process_id, bool print_original, bool print_answer) {
  if (process_id == 0) {
    if (print_original) {
      std::cout << "Original matrix:\n";
      PrintMatrix(matrix.data(), dim);
    }

    printf("Elapsed time: %.4lf seconds.\n", elapsed_time);

    if (print_answer) {
      std::cout << "Resulting matrix:\n";
      PrintMatrix(matrix.data(), dim);
    }
  }
}
