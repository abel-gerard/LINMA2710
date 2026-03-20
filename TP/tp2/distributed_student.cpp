// Compile: mpicxx -[O1 or O2 or O3] distributed.cpp -o distributed
// Run:     mpirun -np [NUM_OF_PROCESSES] ./distributed

#include <mpi.h>

#include <cstdio>
#include <vector>
#include <random>
#include <ctime>
 
#define ROOT_PROC 0

static void compute_bound_local(const std::vector<double> &x_local, std::vector<double> &y_local) {
	const int n_local = static_cast<int>(x_local.size());
	for (int i = 0; i < n_local; ++i) {
		double v = x_local[i];
		for (int k = 0; k < 1; ++k) {
			v = v * v;
		}
		y_local[i] = v;
	}
}

int main(int argc, char **argv) {

	//////////////////////////////////////////////////////////////
    // Initialization of MPI environment and process information
	MPI_Init(&argc, &argv);
	
	// From now on, you are on a specific MPI process, 
	// with its own id (procid) and the total number of processes (nprocs).
	int nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	int procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);

	// Get and print processor name for each process.
	// This is just to show that processes may be running on different nodes.
	int name_length = MPI_MAX_PROCESSOR_NAME;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(proc_name, &name_length);
	std::printf("Process %d/%d is running on node <<%s>>\n", procid, nprocs, proc_name);

	const int N = 20000000;

	/////////////////////////////

	int sendcounts[nprocs];
	int displs[nprocs];
	
	int base_size_per_proc = N / nprocs;
	int remainder_size = N % nprocs;
	
	for (int i = 0; i < nprocs; i++) {
		sendcounts[i] = base_size_per_proc + (i < remainder_size);
		displs[i] = (i == ROOT_PROC) ? 0 : displs[i-1] + sendcounts[i-1];
	}
	
	int count_local = sendcounts[procid];
	std::vector<double> x_local(count_local);
	std::vector<double> y_local(count_local);
	
	std::vector<double> x, y;
	if (procid == ROOT_PROC) {
		x.resize(N);
		y.resize(N);
		std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
		std::normal_distribution<double> dist(0.0, 1.0);
		for (int i = 0; i < N; i++) x[i] = dist(rng);
	}
	
	MPI_Scatterv(
		(procid == ROOT_PROC) ? x.data() : nullptr,
		sendcounts,
		displs,
		MPI_DOUBLE,
		x_local.data(),
		count_local,
		MPI_DOUBLE,
		ROOT_PROC,
		MPI_COMM_WORLD
	);

	compute_bound_local(x_local, y_local);	

	MPI_Gatherv(
		y_local.data(),
		count_local,
		MPI_DOUBLE,
		y.data(),
		sendcounts,
		displs,
		MPI_DOUBLE,
		ROOT_PROC,
		MPI_COMM_WORLD
	);

	if (procid == ROOT_PROC) {
		std::printf("\nCompleted on rank 0.\n");
		std::printf("x[123] = %.17g -> y[123] = %.17g\n", x[123], y[123]);
		std::printf("x[N-1] = %.17g -> y[N-1] = %.17g\n", x[N - 1], y[N - 1]);
	}

	MPI_Finalize();
	return 0;
}
