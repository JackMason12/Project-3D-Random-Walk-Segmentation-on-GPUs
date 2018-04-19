import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.csrilu02Info;
import jcuda.jcusparse.csrsv2Info;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparseSolveAnalysisInfo;
import jcuda.jcublas.*;

import static jcuda.runtime.cudaMemcpyKind.*;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_NO_LEVEL;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_USE_LEVEL;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_LOWER;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_UPPER;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_UNIT;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_DEVICE;
import static jcuda.jcusparse.cusparseStatus.*;
import static jcuda.jcublas.JCublas2.*;

public class MatrixUtils {
	
	public static ColorMap cmap;
	
	public static String PointerContents(Pointer ptr, int count, boolean i) { //gets the contents of a pointer as a string (used for debugging)
		if (i) {
			int[] contents = new int[count];
			cudaMemcpy(Pointer.to(contents), ptr, count*Sizeof.INT, cudaMemcpyDeviceToHost);	
			String out = "";
			for (int j = 0; j < count; j++) {
				out += contents[j];
				out += " ";
			}
			return out;		
		} else {
			double[] contents = new double[count];
			cudaMemcpy(Pointer.to(contents), ptr, count*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			String out = "";
			for (int j = 0; j < count; j++) {
				out += contents[j];
				out += " ";
			}
			return out;
		}		
	}
	
	public static double[] LuSolve(cusparseHandle handle, cusparseMatDescr descrA, int m, int n, int nnz, Pointer AcsrRowIndex_gpuPtr, Pointer AcooColIndex_gpuPtr, Pointer AcooVal_gpuPtr, denseVector b_gpuPtr, boolean iLuBiCGStabSolve) {

		// solve A*x = b
		// b is vector provided
		// x is output probabilities
		// A is sparse matrix defined by variables before the b pointer
		// see paper by Nvidia using cublas1
		// http://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
		// this was also useful:
		// //https://www.cfd-online.com/Wiki/Sample_code_for_BiCGSTAB_-_Fortran_90
		
		// some useful constants
		double[] one_host = { 1.f };
		double[] zero_host = { 0.f };
		double[] minus_one_host = { -1.f };

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
		cudaMalloc(x_gpuPtr, m * Sizeof.DOUBLE);
		cudaMalloc(z_gpuPtr, m * Sizeof.DOUBLE);

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
		cudaMalloc(iLUcooVal_gpuPtr, nnz * Sizeof.DOUBLE);
		cudaMemcpy(iLUcsrRowIndex_gpuPtr, AcsrRowIndex_gpuPtr, (m + 1)
				* Sizeof.INT, cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooColIndex_gpuPtr, AcooColIndex_gpuPtr,
				nnz * Sizeof.INT, cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooVal_gpuPtr, AcooVal_gpuPtr, nnz * Sizeof.DOUBLE,
				cudaMemcpyDeviceToDevice);

		// set up buffer
		int[] pBufferSize_iLU = new int[1];
		int[] pBufferSize_L = new int[1];
		int[] pBufferSize_U = new int[1];
		int pBufferSize;
		Pointer pBuffer = new Pointer();

		// step 3: query how much memory used in csrilu02 and csrsv2, and
		// allocate the buffer
		cusparseDcsrilu02_bufferSize(handle, m, nnz, descr_iLU, AcooVal_gpuPtr,
				AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_iLU,
				pBufferSize_iLU);
		cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz, descr_L,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr,
				info_L, pBufferSize_L);
		cusparseDcsrsv2_bufferSize(handle, trans_U, m, nnz, descr_U,
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

		cusparseDcsrilu02_analysis(handle, m, nnz, descr_iLU, AcooVal_gpuPtr,
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

		cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr,
				info_L, policy_L, pBuffer);

		cusparseDcsrsv2_analysis(handle, trans_U, m, nnz, descr_U,
				AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr,
				info_U, policy_U, pBuffer);

		// step 5: M = L * U
		cusparseDcsrilu02(handle, m, nnz, descr_iLU, iLUcooVal_gpuPtr,
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
		cusparseDcsrsv2_solve(handle, trans_L, m, nnz, Pointer.to(one_host),
				descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
				iLUcooColIndex_gpuPtr, info_L, b_gpuPtr.getPtr(), z_gpuPtr,
				policy_L, pBuffer);

		// step 7: solve U*y = z
		cusparseDcsrsv2_solve(handle, trans_U, m, nnz, Pointer.to(one_host),
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

			cudaMalloc(p, m * Sizeof.DOUBLE);
			cudaMalloc(ph, m * Sizeof.DOUBLE);
			cudaMalloc(q, m * Sizeof.DOUBLE);
			cudaMalloc(r, m * Sizeof.DOUBLE);
			cudaMalloc(rw, m * Sizeof.DOUBLE);
			cudaMalloc(s, m * Sizeof.DOUBLE);
			cudaMalloc(t, m * Sizeof.DOUBLE);

			// BiCGStab parameters (all on host)
			double[] nrmr0 = new double[1];
			double[] nrmr = new double[1];

			double[] rho = { 1.f };
			double[] rhop = new double[1];
			double[] alpha = { 1.f };
			double[] beta = { 0.1f };
			double[] omega = { 1.f };
			double[] temp = new double[1];
			double[] temp2 = new double[1];

			double[] double_host = new double[1]; // used as helper variable to
												// pass doubles

			// BiCGStab numerical parameters
			int maxit = 1000; // maximum number of iterations
			double tol = 1e-3f; // tolerance nrmr / nrmr0[0], which is size of
								// current errors divided by initial error

			// create the info and analyse the lower and upper triangular
			// factors
			cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
			cusparseCreateSolveAnalysisInfo(infoL);
			cusparseSolveAnalysisInfo infoU = new cusparseSolveAnalysisInfo();
			cusparseCreateSolveAnalysisInfo(infoU);

			cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					n, nnz, descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
					iLUcooColIndex_gpuPtr, infoL);
			cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					n, nnz, descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
					iLUcooColIndex_gpuPtr, infoU);

			// 1 : compute initial residual r = b − A x0 ( using initial guess in
			// x )
			cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
					Pointer.to(one_host), descrA, AcooVal_gpuPtr,
					AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, x_gpuPtr,
					Pointer.to(zero_host), r);
			cublasDscal(cublashandle, n, Pointer.to(minus_one_host), r, 1);
			cublasDaxpy(cublashandle, n, Pointer.to(one_host),
					b_gpuPtr.getPtr(), 1, r, 1);

			// 2 : Set p=r and \tilde{r}=r
			cublasDcopy(cublashandle, n, r, 1, p, 1);
			cublasDcopy(cublashandle, n, r, 1, rw, 1);
			cublasDnrm2(cublashandle, n, r, 1, Pointer.to(nrmr0));

			// 3 : repeat until convergence (based on maximum number of
			// iterations and relative residual)

			for (int i = 0; i < maxit; i++) {

				System.out.println("Iteration " + i);

				// 4 : \rho = \tilde{ r }ˆ{T} r
				rhop[0] = rho[0];

				cublasDdot(cublashandle, n, rw, 1, r, 1, Pointer.to(rho));

				if (i > 0) {
					// 1 2 : \beta = (\rho{ i } / \rho { i − 1}) ( \alpha /
					// \omega )
					beta[0] = (rho[0] / rhop[0]) * (alpha[0] / omega[0]);

					// 1 3 : p = r + \beta ( p − \omega v )

					double_host[0] = -omega[0];
					cublasDaxpy(cublashandle, n, Pointer.to(double_host), q, 1,
							p, 1);
					cublasDscal(cublashandle, n, Pointer.to(beta), p, 1);
					cublasDaxpy(cublashandle, n, Pointer.to(one_host), r, 1, p,
							1);
				}

				// 1 5 : A \ hat{p} = p ( sparse lower and upper triangular
				// solves )
				cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, p,
						t);

				cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
						ph);

				// 1 6 : q = A \ hat{p} ( sparse matrix−vector multiplication )
				cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n,
						nnz, Pointer.to(one_host), descrA, AcooVal_gpuPtr,
						AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, ph,
						Pointer.to(zero_host), q);

				// 1 7 : \alpha = \rho_{ i } / ( \tilde{ r }ˆ{T} q )

				jcuda.jcublas.JCublas2.cublasDdot(cublashandle, n, rw, 1, q, 1,
						Pointer.to(temp));

				alpha[0] = rho[0] / temp[0];

				// 1 8 : s = r − \alpha q

				double_host[0] = -alpha[0];
				cublasDaxpy(cublashandle, n, Pointer.to(double_host), q, 1, r, 1);

				// 1 9 : x = x + \alpha \ hat{p};

				cublasDaxpy(cublashandle, n, Pointer.to(alpha), ph, 1,
						x_gpuPtr, 1);

				// 2 0 : check for convergence

				cublasDnrm2(cublashandle, n, r, 1, Pointer.to(nrmr));

				if (nrmr[0] / nrmr0[0] < tol) {
					break;
				}
				// 2 3 : M \ hat{ s } = r ( sparse lower and upper triangular
				// solves )

				cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, r,
						t);

				cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						n, Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
						s);

				// 2 4 : t = A \ hat{ s } ( sparse matrix−vector multiplication
				// )

				cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n,
						nnz, Pointer.to(one_host), descrA, AcooVal_gpuPtr,
						AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, s,
						Pointer.to(zero_host), t);

				// 2 5 : \omega = ( tˆ{T} s ) / ( tˆ{T} t )

				cublasDdot(cublashandle, n, t, 1, r, 1, Pointer.to(temp));
				cublasDdot(cublashandle, n, t, 1, t, 1, Pointer.to(temp2));

				omega[0] = temp[0] / temp2[0];

				// 2 6 : x = x + \omega \ hat{ s }

				cublasDaxpy(cublashandle, n, Pointer.to(omega), s, 1, x_gpuPtr,
						1);

				// cudaMemcpy(Pointer.to(result_host), t, 100*Sizeof.DOUBLE,
				// cudaMemcpyDeviceToHost); //copy results back
				// for(int ii=0;ii<100;ii++)
				// System.out.println("Here t "+ii +"  "+result_host[ii]);

				// 2 7 : r = s − \omega t

				double_host[0] = -omega[0];
				cublasDaxpy(cublashandle, n, Pointer.to(double_host), t, 1, r, 1);

				// check for convergence

				cublasDnrm2(cublashandle, n, r, 1, Pointer.to(nrmr));

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
		double result_host[] = new double[m]; // array to hold results

		cudaMemcpy(Pointer.to(result_host), x_gpuPtr, m * Sizeof.DOUBLE,
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
	
	public static double[][] AddSeeds(double[][] probs, int[] seeds, int[] labels) {
		/*
		 * Add the seeds back to the probabilites in the right positions
		 */
		int label_count = probs.length; //number of labels
		int seed_count = seeds.length;
		int unseeded_count = probs[0].length;
		double[][] probs_fixed = new double[label_count][unseeded_count+seed_count];
		for (int i = 0; i < seed_count; i++) {
			for (int j = 0; j < label_count; j++) {
				if (labels[i] == j) { //if the seed has this label
					probs_fixed[j][seeds[i]] = 1.0; //we know probability is 1
					//System.out.println("seed entered");
				} else {
					probs_fixed[j][seeds[i]] = 0; //otherwise its 0
				}
			}
		}
		boolean all_zero = true;
		int ix = 0;
		int seeds_passed = 0;
		for (int i = 0; i < unseeded_count+seed_count; i++) {
			all_zero = true;
			for (int j = 0; j < label_count; j++) {
				if (probs_fixed[j][i] != 0) all_zero = false;
			}
			if (all_zero == true) {
				for (int j = 0; j < label_count; j++) {
					probs_fixed[j][i] = probs[j][i-seeds_passed];
				}
			} else {
				seeds_passed++;
			}
		}
		return probs_fixed;
	}
	
	public static int[][] GetMask(double[][] probs, int h, int w, int label_count) {
		/*
		 * Get a mask with the same dimensions as the image.
		 * This is obviously temporary, and we would employ a 1d masking solution
		 * Followed by reshaping to the correct dimensions to allow 3 dimensional support
		 * As stated a lot, the methods itself is independent of the dimensionality of the input/output
		 */
		
		int[] out_flat = new int[h*w];
		double largest_prob = 0;
		int largest_ix = 0;
		for (int i = 0; i < h*w; i++) {
			largest_prob = 0;
			for (int j = 0; j < label_count; j++) {
				if (probs[j][i] > largest_prob) {
					largest_prob = probs[j][i];
					largest_ix = j;
				}
			}
			out_flat[i] = largest_ix;
		}
		int ix = 0;
		int[][] out = new int[h][w];
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				out[i][j] = out_flat[ix++];
			}
		}
		//for each row (h*w rows)
		//get largest column (label_count columns)
		//add largest column to out
		//reshape out to h/w
		return out;
	}
	
	public static BufferedImage GetSeedImage(BufferedImage original, int[] seeds, int[] labels) {
		/*
		 * gets an image with the seeds marked
		 */
		BufferedImage out = original;
		int x = 0;
		int y = 0;
		for (int i = 0; i < seeds.length; i++) {
			x = seeds[i] - Math.floorDiv(seeds[i], original.getWidth())*original.getWidth();//seeds[i];
			y = Math.floorDiv(seeds[i], original.getWidth());
			//while(x > original.getWidth() - 1) {
				//x = x - original.getWidth();
				//y++;
			//}
			out.setRGB(x, y, MatrixUtils.cmap.map[labels[i]]);
		}
		return out;
	}
	
	public static BufferedImage GetMaskImage(int[][] mask, int label_count) { //produces a mask image
		/*
		 * Get dimensions of the image to create
		 * Check that they aren't 0
		 * Gets an image of the mask
		 */
		int h = mask.length;
		if (h == 0) {
			System.out.println("Mask height == 0");
			return null;
		}
		int w = mask[0].length;
		if (w == 0) {
			System.out.println("Mask width == 0");
			return null;
		}
		
		/*
		 * Create a colour map, a colour for each label
		 * Generate a random colour and make sure it has not been used for a previous label
		 */
		
		MatrixUtils.cmap = new ColorMap(label_count);
				
		/*
		 * set all the colours to be used in our image using the colour map that we just made
		 */		
		BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				out.setRGB(j, i, MatrixUtils.cmap.map[mask[i][j]]);
			}
		}
		
		return out;
	}
	
	
}
