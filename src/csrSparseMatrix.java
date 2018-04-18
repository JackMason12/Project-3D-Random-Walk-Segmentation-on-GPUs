import jcuda.*;
import jcuda.runtime.JCuda;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import jcuda.jcusparse.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseStatus.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_NO_LEVEL;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_USE_LEVEL;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_LOWER;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_UPPER;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_UNIT;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_DEVICE;

import jcuda.jcublas.*;
import static jcuda.jcublas.JCublas2.*;



public class csrSparseMatrix {

	// variables for matrix A
	private cusparseMatDescr descrA = new cusparseMatDescr();

	private int AcooRowIndex_host[];
	private int AcooColIndex_host[];
	private float AcooVal_host[];

	private Pointer AcooRowIndex_gpuPtr = new Pointer();
	private Pointer AcooColIndex_gpuPtr = new Pointer();
	private Pointer AcooVal_gpuPtr = new Pointer();

	private Pointer AcsrRowIndex_gpuPtr = new Pointer();

	private int nnz; // number of non-zero elements
	private int m; // rows, columns; IMPORTANT: we support square matrices only
					// so m,n used very sloppily
	private int n; // columns

	// cusparse variables
	cusparseHandle handle;

	public csrSparseMatrix(cusparseHandle handle, float[][] e, int m, int n) {
		// m: rows, n: columns, nnz: number of non zero elements
		// create sparse matrix in csr format

		this.m = m;
		this.n = n;
		this.handle = handle;

		// according to JCusparseSample
		JCusparse.setExceptionsEnabled(true);
		JCuda.setExceptionsEnabled(true);

		{ // count nnz elements
			nnz = 0;
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					if (e[i][j] != 0)
						++nnz;
		}

		// create matrix in coo format on host
		AcooRowIndex_host = new int[nnz];
		AcooColIndex_host = new int[nnz];
		AcooVal_host = new float[nnz];

		{
			int count = 0;
			float v;
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					if ((v = e[i][j]) != 0) {
						AcooRowIndex_host[count] = i;
						AcooColIndex_host[count] = j;
						AcooVal_host[count++] = v;
					}
		}

		// Allocate GPU memory and copy the matrix and vectors into it
		cudaMalloc(AcooRowIndex_gpuPtr, nnz * Sizeof.INT);
		cudaMalloc(AcooColIndex_gpuPtr, nnz * Sizeof.INT);
		cudaMalloc(AcooVal_gpuPtr, nnz * Sizeof.FLOAT);

		cudaMemcpy(AcooRowIndex_gpuPtr, Pointer.to(AcooRowIndex_host), nnz
				* Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(AcooColIndex_gpuPtr, Pointer.to(AcooColIndex_host), nnz
				* Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(AcooVal_gpuPtr, Pointer.to(AcooVal_host),
				nnz * Sizeof.FLOAT, cudaMemcpyHostToDevice);

		// Create and set up matrix descriptor
		cusparseCreateMatDescr(descrA);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

		// Exercise conversion routines (convert matrix from COO 2 CSR format)
		cudaMalloc(AcsrRowIndex_gpuPtr, (m + 1) * Sizeof.INT);
		cusparseXcoo2csr(handle, AcooRowIndex_gpuPtr, nnz, m,
				AcsrRowIndex_gpuPtr, CUSPARSE_INDEX_BASE_ZERO);

	}

	public float[] LuSolve(denseVector b_gpuPtr) {
		return LuSolve(b_gpuPtr, false);
	}

	public float[] LuSolve(denseVector b_gpuPtr, boolean iLuBiCGStabSolve) {

		// some useful constants
		float[] one_host = { 1.f };
		float[] zero_host = { 0.f };
		float[] minus_one_host = { -1.f };

		// iLU part adapted from nvidia cusparse documentation

		// Suppose that A is m x m sparse matrix represented by CSR format,
		// Assumption
		// - handle is already created by cusparseCreate(),
		// - (AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, AcooVal_gpuPtr) is CSR
		// of A on device memory,
		// - b_gpuPtr is right hand side vector on device memory,
		// - x_gpuPtr is solution vector on device memory.
		// - z_gpuPtr is intermediate result on device memory.

		// setup solution vector and intermediate vector
		Pointer x_gpuPtr = new Pointer();
		Pointer z_gpuPtr = new Pointer();
		cudaMalloc(x_gpuPtr, m * Sizeof.FLOAT);
		cudaMalloc(z_gpuPtr, m * Sizeof.FLOAT);

		// setting up pointers for the sparse iLU matrix, which contains L and U
		// Nvidia's original example overwrites matrix A, which is not ideal
		// when later using iLuBiCGStabSolve
		Pointer iLUcooColIndex_gpuPtr = new Pointer();
		Pointer iLUcooVal_gpuPtr = new Pointer();
		Pointer iLUcsrRowIndex_gpuPtr = new Pointer();

		// step 1: create descriptors/policies/operation modi for iLU, L, and U
		cusparseMatDescr descr_iLU = new cusparseMatDescr();
		cusparseCreateMatDescr(descr_iLU);
		cusparseSetMatIndexBase(descr_iLU, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_iLU, CUSPARSE_MATRIX_TYPE_GENERAL);
		int policy_iLU = CUSPARSE_SOLVE_POLICY_NO_LEVEL;

		cusparseMatDescr descr_L = new cusparseMatDescr();
		;
		cusparseCreateMatDescr(descr_L);
		cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
		cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
		int policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
		int trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;

		cusparseMatDescr descr_U = new cusparseMatDescr();
		;
		cusparseCreateMatDescr(descr_U);
		cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
		cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
		int policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
		int trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

		// step 2: create a empty info structure
		// we need one info for csrilu02 and two info's for csrsv2
		csrilu02Info info_iLU = new csrilu02Info();
		csrsv2Info info_L = new csrsv2Info();
		csrsv2Info info_U = new csrsv2Info();
		cusparseCreateCsrilu02Info(info_iLU);
		cusparseCreateCsrsv2Info(info_L);
		cusparseCreateCsrsv2Info(info_U);

		// copy matrix A into iLU
		cudaMalloc(iLUcsrRowIndex_gpuPtr, (m + 1) * Sizeof.INT);
		cudaMalloc(iLUcooColIndex_gpuPtr, nnz * Sizeof.INT);
		cudaMalloc(iLUcooVal_gpuPtr, nnz * Sizeof.FLOAT);
		cudaMemcpy(iLUcsrRowIndex_gpuPtr, AcsrRowIndex_gpuPtr, (m + 1)
				* Sizeof.INT, cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooColIndex_gpuPtr, AcooColIndex_gpuPtr,
				nnz * Sizeof.INT, cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooVal_gpuPtr, AcooVal_gpuPtr, nnz * Sizeof.FLOAT,
				cudaMemcpyDeviceToDevice);

		// set up buffer
		int[] pBufferSize_iLU = new int[1];
		int[] pBufferSize_L = new int[1];
		int[] pBufferSize_U = new int[1];
		int pBufferSize;
		Pointer pBuffer = new Pointer();

		// step 3: query how much memory used in csrilu02 and csrsv2, and
		// allocate the buffer
		cusparseScsrilu02_bufferSize(handle, m, nnz, descr_iLU, AcooVal_gpuPtr,
				AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_iLU,
				pBufferSize_iLU);
		cusparseScsrsv2_bufferSize(handle, trans_L, m, nnz, descr_L,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr,
				info_L, pBufferSize_L);
		cusparseScsrsv2_bufferSize(handle, trans_U, m, nnz, descr_U,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr,
				info_U, pBufferSize_U);

		pBufferSize = Math.max(pBufferSize_iLU[0],
				Math.max(pBufferSize_L[0], pBufferSize_U[0]));
		// System.out.println("in csrSparseMatrix.LuSolve(),buffersize = "+
		// pBufferSize);

		// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
		cudaMalloc(pBuffer, pBufferSize);

		// step 4: perform analysis of incomplete Cholesky on M
		// perform analysis of triangular solve on L
		// perform analysis of triangular solve on U
		// The lower(upper) triangular part of M has the same sparsity pattern
		// as L(U),
		// we can do analysis of csrilu0 and csrsv2 simultaneously.

		cusparseScsrilu02_analysis(handle, m, nnz, descr_iLU, AcooVal_gpuPtr,
				AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_iLU, policy_iLU,
				pBuffer);

		Pointer structural_zero = new Pointer();
		cudaMalloc(structural_zero, Sizeof.INT);

		// int[] cusparsePointerMode = new int[1];
		// default mode seems to be HOST
		// cusparseGetPointerMode(handle, cusparsePointerMode);
		// System.out.printf("Cusparse pointer mode %d \n",
		// cusparsePointerMode[0]);
		// we need to switch to DEVICE before using cusparseXcsrilu02_zeroPivot,
		// for obscure reasons, and switch back to HOST afterwards
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
		if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(handle,
				info_iLU, structural_zero)) {
			int[] sz = new int[1];
			cudaMemcpy(Pointer.to(sz), structural_zero, Sizeof.INT,
					cudaMemcpyDeviceToHost); // copy results back
			System.out.printf("A(%d,%d) is missing\n", sz[0], sz[0]);
		}
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

		cusparseScsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr,
				info_L, policy_L, pBuffer);

		cusparseScsrsv2_analysis(handle, trans_U, m, nnz, descr_U,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr,
				info_U, policy_U, pBuffer);

		// step 5: M = L * U
		cusparseScsrilu02(handle, m, nnz, descr_iLU, iLUcooVal_gpuPtr,
				iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_iLU,
				policy_iLU, pBuffer);

		Pointer numerical_zero = new Pointer();
		cudaMalloc(numerical_zero, Sizeof.INT);

		// same trick of switching modes needed here
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
		if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(handle,
				info_iLU, numerical_zero)) {
			int[] nz = new int[1];
			cudaMemcpy(Pointer.to(nz), numerical_zero, Sizeof.INT,
					cudaMemcpyDeviceToHost); // copy results back
			System.out.printf("U(%d,%d) is zero\n", nz[0], nz[0]);
		}
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

		// step 6: solve L*z = x
		cusparseScsrsv2_solve(handle, trans_L, m, nnz, Pointer.to(one_host),
				descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
				iLUcooColIndex_gpuPtr, info_L, b_gpuPtr.getPtr(), z_gpuPtr,
				policy_L, pBuffer);

		// step 7: solve U*y = z
		cusparseScsrsv2_solve(handle, trans_U, m, nnz, Pointer.to(one_host),
				descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
				iLUcooColIndex_gpuPtr, info_U, z_gpuPtr, x_gpuPtr, policy_U,
				pBuffer);

		// CG routine
		if (iLuBiCGStabSolve) {

			// see paper by Nvidia using cublas1
			// http://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
			// this was also useful:
			// //https://www.cfd-online.com/Wiki/Sample_code_for_BiCGSTAB_-_Fortran_90

			// we make extensive use of JCublas2
			cublasHandle cublashandle = new cublasHandle();
			jcuda.jcublas.JCublas2.cublasCreate(cublashandle);

			// /*****BiCGStabCode*****/
			// /*ASSUMPTIONS:
			// 1.The CUSPARSE and CUBLAS libraries have been initialized.
			// 2.The appropriate memory has been allocated and set to zero.
			// 3.The matrixA (valA, csrRowPtrA, csrColIndA) and the incomplete−
			// LUlowerL (valL, csrRowPtrL, csrColIndL) and upperU (valU,
			// csrRowPtrU, csrColIndU) triangular factors have been
			// computed and are present in the device (GPU) memory.*/
			//

			// the above requirements are met

			// we create a number of pointers according to the method, and
			// subsequently allocate memory
			// TODO: rename these according to _gpuPtr scheme
			Pointer p = new Pointer();
			Pointer ph = new Pointer();
			Pointer q = new Pointer();
			Pointer r = new Pointer();
			Pointer rw = new Pointer();
			Pointer s = new Pointer();
			Pointer t = new Pointer();

			cudaMalloc(p, m * Sizeof.FLOAT);
			cudaMalloc(ph, m * Sizeof.FLOAT);
			cudaMalloc(q, m * Sizeof.FLOAT);
			cudaMalloc(r, m * Sizeof.FLOAT);
			cudaMalloc(rw, m * Sizeof.FLOAT);
			cudaMalloc(s, m * Sizeof.FLOAT);
			cudaMalloc(t, m * Sizeof.FLOAT);

			// BiCGStab parameters (all on host)
			float[] nrmr0 = new float[1];
			float[] nrmr = new float[1];

			float[] rho = { 1.f };
			float[] rhop = new float[1];
			float[] alpha = { 1.f };
			float[] beta = { 0.1f };
			float[] omega = { 1.f };
			float[] temp = new float[1];
			float[] temp2 = new float[1];

			float[] float_host = new float[1]; // used as helper variable to
												// pass floats

			// BiCGStab numerical parameters
			int maxit = 1000; // maximum number of iterations
			float tol = 1e-3f; // tolerance nrmr / nrmr0[0], which is size of
								// current errors divided by initial error

			// create the info and analyse the lower and upper triangular
			// factors
			cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
			cusparseCreateSolveAnalysisInfo(infoL);
			cusparseSolveAnalysisInfo infoU = new cusparseSolveAnalysisInfo();
			cusparseCreateSolveAnalysisInfo(infoU);

			cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					n, nnz, descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
					iLUcooColIndex_gpuPtr, infoL);
			cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					n, nnz, descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
					iLUcooColIndex_gpuPtr, infoU);

			// 1 : compute initial residual r = b − A x0 ( using initial guess in
			// x )
			cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
					Pointer.to(one_host), descrA, AcooVal_gpuPtr,
					AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, x_gpuPtr,
					Pointer.to(zero_host), r);
			cublasSscal(cublashandle, n, Pointer.to(minus_one_host), r, 1);
			cublasSaxpy(cublashandle, n, Pointer.to(one_host),
					b_gpuPtr.getPtr(), 1, r, 1);

			// 2 : Set p=r and \tilde{r}=r
			cublasScopy(cublashandle, n, r, 1, p, 1);
			cublasScopy(cublashandle, n, r, 1, rw, 1);
			cublasSnrm2(cublashandle, n, r, 1, Pointer.to(nrmr0));

			// 3 : repeat until convergence (based on maximum number of
			// iterations and relative residual)

			for (int i = 0; i < maxit; i++) {

				System.out.println("Iteration " + i);

				// 4 : \rho = \tilde{ r }ˆ{T} r
				rhop[0] = rho[0];

				cublasSdot(cublashandle, n, rw, 1, r, 1, Pointer.to(rho));

				if (i > 0) {
					// 1 2 : \beta = (\rho{ i } / \rho { i − 1}) ( \alpha /
					// \omega )
					beta[0] = (rho[0] / rhop[0]) * (alpha[0] / omega[0]);

					// 1 3 : p = r + \beta ( p − \omega v )

					float_host[0] = -omega[0];
					cublasSaxpy(cublashandle, n, Pointer.to(float_host), q, 1,
							p, 1);
					cublasSscal(cublashandle, n, Pointer.to(beta), p, 1);
					cublasSaxpy(cublashandle, n, Pointer.to(one_host), r, 1, p,
							1);
				}

				// 1 5 : A \ hat{p} = p ( sparse lower and upper triangular
				// solves )
				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, p,
						t);

				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
						ph);

				// 1 6 : q = A \ hat{p} ( sparse matrix−vector multiplication )
				cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n,
						nnz, Pointer.to(one_host), descrA, AcooVal_gpuPtr,
						AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, ph,
						Pointer.to(zero_host), q);

				// 1 7 : \alpha = \rho_{ i } / ( \tilde{ r }ˆ{T} q )

				jcuda.jcublas.JCublas2.cublasSdot(cublashandle, n, rw, 1, q, 1,
						Pointer.to(temp));

				alpha[0] = rho[0] / temp[0];

				// 1 8 : s = r − \alpha q

				float_host[0] = -alpha[0];
				cublasSaxpy(cublashandle, n, Pointer.to(float_host), q, 1, r, 1);

				// 1 9 : x = x + \alpha \ hat{p};

				cublasSaxpy(cublashandle, n, Pointer.to(alpha), ph, 1,
						x_gpuPtr, 1);

				// 2 0 : check for convergence

				cublasSnrm2(cublashandle, n, r, 1, Pointer.to(nrmr));

				if (nrmr[0] / nrmr0[0] < tol) {
					break;
				}
				// 2 3 : M \ hat{ s } = r ( sparse lower and upper triangular
				// solves )

				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, r,
						t);

				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
						s);

				// 2 4 : t = A \ hat{ s } ( sparse matrix−vector multiplication
				// )

				cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n,
						nnz, Pointer.to(one_host), descrA, AcooVal_gpuPtr,
						AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, s,
						Pointer.to(zero_host), t);

				// 2 5 : \omega = ( tˆ{T} s ) / ( tˆ{T} t )

				cublasSdot(cublashandle, n, t, 1, r, 1, Pointer.to(temp));
				cublasSdot(cublashandle, n, t, 1, t, 1, Pointer.to(temp2));

				omega[0] = temp[0] / temp2[0];

				// 2 6 : x = x + \omega \ hat{ s }

				cublasSaxpy(cublashandle, n, Pointer.to(omega), s, 1, x_gpuPtr,
						1);

				// cudaMemcpy(Pointer.to(result_host), t, 100*Sizeof.FLOAT,
				// cudaMemcpyDeviceToHost); //copy results back
				// for(int ii=0;ii<100;ii++)
				// System.out.println("Here t "+ii +"  "+result_host[ii]);

				// 2 7 : r = s − \omega t

				float_host[0] = -omega[0];
				cublasSaxpy(cublashandle, n, Pointer.to(float_host), t, 1, r, 1);

				// check for convergence

				cublasSnrm2(cublashandle, n, r, 1, Pointer.to(nrmr));

				if (nrmr[0] / nrmr0[0] < tol) {
					break;
				}

				System.out.println("nrmr: " + nrmr[0] + " nrmr0: " + nrmr0[0]
						+ " alpha: " + alpha[0] + " beta: " + beta[0]
						+ " rho: " + rho[0] + " temp: " + temp[0] + " temp2: "
						+ temp2[0] + " omega: " + omega[0]);

			}

			cudaFree(p);
			cudaFree(ph);
			cudaFree(q);
			cudaFree(r);
			cudaFree(rw);
			cudaFree(s);
			cudaFree(t);

			cusparseDestroySolveAnalysisInfo(infoL);
			cusparseDestroySolveAnalysisInfo(infoU);

			cublasDestroy(cublashandle);

		} // CG routine
			// /needs changing
		float result_host[] = new float[m]; // array to hold results

		cudaMemcpy(Pointer.to(result_host), x_gpuPtr, m * Sizeof.FLOAT,
				cudaMemcpyDeviceToHost); // copy results back

		cudaFree(x_gpuPtr);
		cudaFree(z_gpuPtr);
		cudaFree(iLUcooColIndex_gpuPtr);
		cudaFree(iLUcooVal_gpuPtr);
		cudaFree(iLUcsrRowIndex_gpuPtr);
		cudaFree(pBuffer);
		cudaFree(structural_zero);
		cudaFree(numerical_zero);

		cusparseDestroyMatDescr(descr_iLU);
		cusparseDestroyMatDescr(descr_L);
		cusparseDestroyMatDescr(descr_U);
		cusparseDestroyCsrilu02Info(info_iLU);
		cusparseDestroyCsrsv2Info(info_L);
		cusparseDestroyCsrsv2Info(info_U);

		return result_host;

	}

	// ///////////////////////////////////////////////////
	public float[] mldivide(denseVector b_gpuPtr) {

		// solves triangular systems

		float result_host[] = new float[n]; // array to hold results on host
		Pointer result_gpuPtr = new Pointer();
		cudaMalloc(result_gpuPtr, n * Sizeof.FLOAT); // allocate gpu memory for
														// results

		float[] one_host = { 1.f };

		cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo();
		cusparseCreateSolveAnalysisInfo(info);

		int cusparseStatus;

		cusparseStatus = cusparseScsrsm_analysis(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, descrA,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info);

		if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
			System.out
					.println("in csrSparseMatrix.mldivide(), problem with cusparseScsrsm_analysis, cusparseStatus = "
							+ cusparseStatus);

		cusparseStatus = cusparseScsrsm_solve(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, m, 1, Pointer.to(one_host),
				descrA, AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
				AcooColIndex_gpuPtr, info, b_gpuPtr.getPtr(), n, result_gpuPtr,
				n);

		if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
			System.out
					.println("in csrSparseMatrix.mldivide(), problem with cusparseScsrsm_solve, cusparseStatus = "
							+ cusparseStatus);

		// copy results back to host
		cudaMemcpy(Pointer.to(result_host), result_gpuPtr, n * Sizeof.FLOAT,
				cudaMemcpyDeviceToHost);

		JCuda.cudaDeviceSynchronize();

		return result_host;
	}

	// ///////////////////////////////////////////////////
	public void free() {
		// clear Matrix A
		cudaFree(AcsrRowIndex_gpuPtr);
		cudaFree(AcooRowIndex_gpuPtr);
		cudaFree(AcooColIndex_gpuPtr);
		cudaFree(AcooVal_gpuPtr);
	}

	// ///////////////////////////////////////////////////
	public void print_coo() {

		System.out.printf(" csrSparseMatrix in COO format:\n");
		for (int i = 0; i < nnz; i++) {
			System.out.printf("cooRowInded_host[%d]=%d  ", i,
					AcooRowIndex_host[i]);
			System.out.printf("cooColInded_host[%d]=%d  ", i,
					AcooColIndex_host[i]);
			System.out.printf("AcooVal_host[%d]=%f     \n", i, AcooVal_host[i]);
		}

	}

	// ///////////////////////////////////////////////////

} // class csrSparseMatrix
