import java.awt.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparseSolveAnalysisInfo;
import jcuda.runtime.JCuda;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;

public class fix {
	static double[][] getProbabilities(int[] pixels, int pixelCount, Edge[] edges, int edgeCount, double beta, int[] seeds, int seedCount, int[] labels, int labelCount) {
		System.out.println("hi");
		JCusparse.setExceptionsEnabled(true);
		JCuda.setExceptionsEnabled(true);
		cusparseHandle handle = new cusparseHandle(); //handle and cusparse init stuff
		cusparseCreate(handle);
		
		
		//create all of our matrix descriptions
		cusparseMatDescr descrA = new cusparseMatDescr();  //description for A
		cusparseMatDescr descrAt = new cusparseMatDescr(); //description for A^T
		cusparseMatDescr descrC = new cusparseMatDescr(); //description for C
		cusparseMatDescr descrAtC = new cusparseMatDescr(); //description for A^TC
		cusparseMatDescr descrLap = new cusparseMatDescr(); //description for Laplacian
		cusparseMatDescr descrLu = new cusparseMatDescr(); //description for unseeded part of laplacian
		cusparseMatDescr descrLb = new cusparseMatDescr(); //description for boundary values of laplacian
		cusparseMatDescr descrB = new cusparseMatDescr(); //description for B
		cusparseMatDescr descrRHS = new cusparseMatDescr(); //description for RHS of equation
		
		cusparseCreateMatDescr(descrA);
		cusparseCreateMatDescr(descrAt);
		cusparseCreateMatDescr(descrC);
		cusparseCreateMatDescr(descrAtC);
		cusparseCreateMatDescr(descrLap);
		cusparseCreateMatDescr(descrLu);
		cusparseCreateMatDescr(descrLb);
		cusparseCreateMatDescr(descrB);
		cusparseCreateMatDescr(descrRHS);
		
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrAt, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrAtC, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrLap, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrLu, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrLb, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrRHS, CUSPARSE_MATRIX_TYPE_GENERAL);
		
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrAt, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrAtC, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrLap, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrLu, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrLb, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrRHS, CUSPARSE_INDEX_BASE_ZERO);
		
		System.out.println("Matrix Descriptions created");
		
		int nnzA = edgeCount*2; //number of non zero elements of a
		int nnzAt = nnzA; //number of non zero elements of A^T
		int nnzC = edgeCount;
		
		int mA = edgeCount; //height of matrix A
		int mAt = pixelCount; //height of matrix A^T
		int mC = edgeCount;
		
		int nA = pixelCount;
		int nC = edgeCount;

		int ARowCOO[] = new int[nnzA]; //arrays for holding A in Coo form
		int AColCOO[] = new int[nnzA];
		double AValCOO[] = new double[nnzA];
		
		int AtRowCOO[] = AColCOO; //set arrays for holding A^T in coo form
		int AtColCOO[] = ARowCOO;
		double AtValCOO[] = AValCOO;
		
		ArrayList<Integer> AtRowCOOList = new ArrayList<Integer>(); //lists to hold A^T (so we can sort them)
		ArrayList<Integer> AtColCOOList = new ArrayList<Integer>();
		ArrayList<Double> AtValCOOList = new ArrayList<Double>();
		
		int CRowCOO[] = new int[nnzC]; //arrays to hold C in COO form
		int CColCOO[] = new int[nnzC];
		double CValCOO[] = new double[nnzC];

		int AEntries = 0; //number of entries into matrix a

		for (int i = 0; i < edgeCount; i++) { //populate coo arrays in row-major format
			ARowCOO[AEntries] = edges[i].start;
			AColCOO[AEntries] = i;
			AValCOO[AEntries++] = -1;
			ARowCOO[AEntries] = edges[i].end;
			AColCOO[AEntries] = i;
			AValCOO[AEntries++] = 1;
		}

		AEntries = 0; //zero unneeded memory		

		for (int i = 0; i < nnzA; i++) { //add every element to the lists
			AtRowCOOList.add(AtRowCOO[i]);
			AtColCOOList.add(AtColCOO[i]);
			AtValCOOList.add(AtValCOO[i]);
		}

		concurrentSort(AtColCOOList, AtColCOOList, AtRowCOOList, AtValCOOList); //sort the lists so we have row-major format for transpose

		for (int i = 0; i < nnzA; i++) { //put the list elements back in the arrays
			AtRowCOO[i] = AtRowCOOList.get(i);
			AtColCOO[i] = AtColCOOList.get(i);
			AtValCOO[i] = AtValCOOList.get(i);
		}

		AtRowCOOList = null; //null out unneeded memory
		AtColCOOList = null;
		AtValCOOList = null;

		for (int i = 0; i < nnzC; i++) { //populate arrays to hold C
			CRowCOO[i] = i;
			CColCOO[i] = i;
			CValCOO[i] = weight(pixels[edges[i].start], pixels[edges[i].end], beta);
			System.out.printf("Row: %d, Col: %d, Val: %f\n", i, i, weight(pixels[edges[i].start], pixels[edges[i].end], beta));
		}

		Pointer ARowCOOPtr = new Pointer(); //declare pointers to hold on GPU
		Pointer ARowCSRPtr = new Pointer();
		Pointer AColCSRPtr = new Pointer();
		Pointer AValCSRPtr = new Pointer();
		
		Pointer AtRowCOOPtr = new Pointer();
		Pointer AtRowCSRPtr = new Pointer();
		Pointer AtColCSRPtr = new Pointer();
		Pointer AtValCSRPtr = new Pointer();
		
		Pointer CRowCOOPtr = new Pointer();
		Pointer CRowCSRPtr = new Pointer();
		Pointer CColCSRPtr = new Pointer();
		Pointer CValCSRPtr = new Pointer();

		cudaMalloc(ARowCOOPtr, nnzA*Sizeof.INT); //allocate memory to pointers
		cudaMalloc(ARowCSRPtr, (mA+1)*Sizeof.INT);
		cudaMalloc(AColCSRPtr, nnzA*Sizeof.INT);
		cudaMalloc(AValCSRPtr, nnzA*Sizeof.DOUBLE);
		
		cudaMalloc(AtRowCOOPtr, nnzAt*Sizeof.INT);
		cudaMalloc(AtRowCSRPtr, (mAt+1)*Sizeof.INT);
		cudaMalloc(AtColCSRPtr, nnzAt*Sizeof.INT);
		cudaMalloc(AtValCSRPtr, nnzAt*Sizeof.DOUBLE);
		
		cudaMalloc(CRowCOOPtr, nnzC*Sizeof.INT);
		cudaMalloc(CRowCSRPtr, (mC+1)*Sizeof.INT);
		cudaMalloc(CColCSRPtr, nnzC*Sizeof.INT);
		cudaMalloc(CValCSRPtr, nnzC*Sizeof.DOUBLE);
		
		JCuda.cudaDeviceSynchronize();

		cudaMemcpy(ARowCOOPtr, Pointer.to(ARowCOO), nnzA*Sizeof.INT, cudaMemcpyHostToDevice); //copy array contents into memory on gpu
		cudaMemcpy(AColCSRPtr, Pointer.to(AColCOO), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(AValCSRPtr, Pointer.to(AValCOO), nnzA*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		
		cudaMemcpy(AtRowCOOPtr, Pointer.to(AtRowCOO), nnzAt*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(AtColCSRPtr, Pointer.to(AtColCOO), nnzAt*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(AtValCSRPtr, Pointer.to(AtValCOO), nnzAt*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		
		cudaMemcpy(CRowCOOPtr, Pointer.to(CRowCOO), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(CColCSRPtr, Pointer.to(CColCOO), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(CValCSRPtr, Pointer.to(CValCOO), nnzC*Sizeof.DOUBLE, cudaMemcpyHostToDevice);

		
		JCuda.cudaDeviceSynchronize();
		

		cusparseXcoo2csr(handle, ARowCOOPtr, nnzA, mA, ARowCSRPtr, CUSPARSE_INDEX_BASE_ZERO); //convert COO row representation to CSR
		
		JCuda.cudaDeviceSynchronize();
		
		cusparseXcoo2csr(handle, AtRowCOOPtr, nnzAt, mAt, AtRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
		
		JCuda.cudaDeviceSynchronize();
		
		//for some reason, this ruins CColCSRPtr completely. (is it allocating the same bit of memory?)
		cusparseXcoo2csr(handle, CRowCOOPtr, nnzC, mC, CRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
		
		JCuda.cudaDeviceSynchronize();
		

		int CRowCSR[] = new int[nnzC];
		cudaMemcpy(Pointer.to(CRowCSR), CColCSRPtr, (nnzC)*Sizeof.INT, cudaMemcpyDeviceToHost);
		for (int i = 0; i < nnzC; i++) {
			System.out.println(CRowCSR[i]);
		}
		
		
		cudaFree(ARowCOOPtr); //free up COO pointers
		cudaFree(AtRowCOOPtr);
		cudaFree(CRowCOOPtr);

		System.out.println("A, A^T and C created");
		
		/* Calculate Laplacian from A^t, A and C
		 */
		int m,n,k; //set up variables for matrix multiplication
		m = mAt;
		n = nC;
		k = mC;
		int mAtC = m;
		int nAtC = n;

		Pointer AtCRowCSRPtr = new Pointer(); //pointers for multiplication
		Pointer AtCColCSRPtr = new Pointer();
		Pointer AtCValCSRPtr = new Pointer();
		Pointer nnzAtCPtr = new Pointer();

		int nnzAtCArr[] = new int[1]; //array to get nnzAtC back
		
		cudaMalloc(nnzAtCPtr, Sizeof.INT); //allocate memory for pointers
		cudaMalloc(AtCRowCSRPtr, (m+1)*Sizeof.INT);


		//get nnz for the multiplication
		cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrAt, nnzAt, AtRowCSRPtr, AtColCSRPtr, descrC, nnzC, CRowCSRPtr, CColCSRPtr, descrAtC, AtCRowCSRPtr, nnzAtCPtr);
		JCuda.cudaDeviceSynchronize();
		cudaMemcpy(Pointer.to(nnzAtCArr), nnzAtCPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
		//int nnzAtC = nnzAtCArr[0];
		//nnzAtCArr = null;
		System.out.println(nnzAtCArr[0]);

		cudaMalloc(AtCColCSRPtr, nnzAtCArr[0]*Sizeof.INT);
		cudaMalloc(AtCValCSRPtr, nnzAtCArr[0]*Sizeof.DOUBLE);
		//perform multiplication to get A^TC
		cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrAt, nnzAt, AtValCSRPtr, AtRowCSRPtr, AtColCSRPtr, descrC, nnzC, CValCSRPtr, CRowCSRPtr, CColCSRPtr, descrAtC, AtCValCSRPtr, AtCRowCSRPtr, AtCColCSRPtr);

		cudaFree(AtRowCSRPtr); //free up At and C (dont need them any more)
		cudaFree(AtColCSRPtr);
		cudaFree(AtValCSRPtr);
		
		cudaFree(CRowCSRPtr);
		cudaFree(CColCSRPtr);
		cudaFree(CValCSRPtr);


		m = mAtC; //set up variables
		n = nA;
		k = nAtC;
		int mLap = m;

		Pointer LapRowCSRPtr = new Pointer(); //pointers for multiplication
		Pointer LapColCSRPtr = new Pointer();
		Pointer LapValCSRPtr = new Pointer();
		Pointer nnzLapPtr = new Pointer();

		cudaMalloc(nnzLapPtr, Sizeof.INT); //allocate memory for pointers
		cudaMalloc(LapRowCSRPtr, (m+1)*Sizeof.INT);

		int nnzLapArr[] = new int[1];
		//get nnz for the multiplication
		cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrAtC, nnzAtCArr[0], AtCRowCSRPtr, AtCColCSRPtr, descrA, nnzA, ARowCSRPtr, AColCSRPtr, descrLap, LapRowCSRPtr, nnzLapPtr);
		cudaMemcpy(Pointer.to(nnzLapArr), nnzLapPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
		int nnzLap = nnzLapArr[0]; //copy nnz into variable
		nnzLapArr = null;

		cudaMalloc(LapColCSRPtr, nnzLap*Sizeof.INT); //allocate memory for the rest of the variables
		cudaMalloc(LapValCSRPtr, nnzLap*Sizeof.DOUBLE);
		//perform last part of multiplication
		cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrAtC, nnzAtCArr[0], AtCValCSRPtr, AtCRowCSRPtr, AtCColCSRPtr, descrA, nnzA, AValCSRPtr, ARowCSRPtr, AColCSRPtr, descrLap, LapValCSRPtr, LapRowCSRPtr, LapColCSRPtr);

		cudaFree(AtCRowCSRPtr); //free up memory for A^TC and A, no longer needed
		cudaFree(AtCColCSRPtr);
		cudaFree(AtCValCSRPtr);
		
		cudaFree(ARowCSRPtr);
		cudaFree(AColCSRPtr);
		cudaFree(AValCSRPtr);

		//convert laplacian back to COO, and get back on device to manipulate it
		Pointer LapRowCOOPtr = new Pointer();

		cusparseXcsr2coo(handle, LapRowCSRPtr, nnzLap, mLap, LapRowCOOPtr, CUSPARSE_INDEX_BASE_ZERO);

		int LapRowCOO[] = new int[nnzLap];
		int LapColCOO[] = new int[nnzLap];
		double LapValCOO[] = new double[nnzLap];

		cudaMemcpy(Pointer.to(LapRowCOO), LapRowCOOPtr, nnzLap*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(LapColCOO), LapColCSRPtr, nnzLap*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(LapValCOO), LapValCSRPtr, nnzLap*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);


		ArrayList<Integer> LuRowCOOList = new ArrayList<Integer>();
		ArrayList<Integer> LuColCOOList = new ArrayList<Integer>();
		ArrayList<Double> LuValCOOList = new ArrayList<Double>();
		ArrayList<Integer> LbRowCOOList = new ArrayList<Integer>();
		ArrayList<Integer> LbColCOOList = new ArrayList<Integer>();
		ArrayList<Double> LbValCOOList = new ArrayList<Double>();

		for (int i = 0; i < nnzLap; i++) {
			System.out.println(LapValCOO[i]);
			LuRowCOOList.add(LapRowCOO[i]);
			LuColCOOList.add(LapColCOO[i]);
			LuValCOOList.add(LapValCOO[i]);
			LbRowCOOList.add(LapRowCOO[i]);
			LbColCOOList.add(LapColCOO[i]);
			LbValCOOList.add(LapValCOO[i]);
		}

		int ix;
		int nnzLu = nnzLap;
		int mLu = pixelCount-seedCount;
		int nnzLb = nnzLap;
		int mLb = pixelCount-seedCount;
		
		ArrayList<Integer> seed = new ArrayList<Integer>();
		for (int i = 0; i < seedCount; i++) {
			seed.add(seeds[i]);
		}

		for (int i = 0; i < pixelCount; i++) {
			if (seed.contains(i)) {
				ix = LuRowCOOList.indexOf(i);
				while (ix != -1) {
					LuRowCOOList.remove(ix);
					LuColCOOList.remove(ix);
					LuValCOOList.remove(ix);
					nnzLu--;
					ix = LuRowCOOList.indexOf(i);
				}
				ix = LuColCOOList.indexOf(i);
				while (ix != -1) {
					LuRowCOOList.remove(ix);
					LuColCOOList.remove(ix);
					LuValCOOList.remove(ix);
					nnzLu--;
					ix = LuColCOOList.indexOf(i);
				}
				ix = LbRowCOOList.indexOf(i);
				while (ix != -1) {
					LbRowCOOList.remove(ix);
					LbColCOOList.remove(ix);
					LbValCOOList.remove(ix);
					nnzLb--;
					ix = LbRowCOOList.indexOf(i);
				}
			} else {
				ix = LbColCOOList.indexOf(i);
				while (ix != -1) {
					LbRowCOOList.remove(ix);
					LbColCOOList.remove(ix);
					LbValCOOList.remove(ix);
					nnzLb--;
					ix = LbColCOOList.indexOf(i);
				}
			}
		}

		int LuRowCOO[] = new int[nnzLu];
		int LuColCOO[] = new int[nnzLu];
		double LuValCOO[] = new double[nnzLu];
		int LbRowCOO[] = new int[nnzLb];
		int LbColCOO[] = new int[nnzLb];
		double LbValCOO[] = new double[nnzLb];

		for (int i = 0; i < nnzLu; i++) {
			LuRowCOO[i] = LuRowCOOList.get(i);
			LuColCOO[i] = LuColCOOList.get(i);
			LuValCOO[i] = LuValCOOList.get(i);
		}

		for (int i = 0; i < nnzLb; i++) {
			LbRowCOO[i] = -LbRowCOOList.get(i);
			LbColCOO[i] = -LbColCOOList.get(i);
			LbValCOO[i] = -LbValCOOList.get(i);
		}

		int nnzB = seedCount;
		int mB = seedCount;
		int BRowCOO[] = new int[nnzB];
		int BColCOO[] = new int[nnzB];
		double BValCOO[] = new double[nnzB];
		for (int i = 0; i < seedCount; i++) {
			BRowCOO[i] = i;
			BColCOO[i] = labels[i];
			BValCOO[i] = 1;
		}

		Pointer LuRowCOOPtr = new Pointer();
		Pointer LuRowCSRPtr = new Pointer();
		Pointer LuColCSRPtr = new Pointer();
		Pointer LuValCSRPtr = new Pointer();

		Pointer LbRowCOOPtr = new Pointer();
		Pointer LbRowCSRPtr = new Pointer();
		Pointer LbColCSRPtr = new Pointer();
		Pointer LbValCSRPtr = new Pointer();

		Pointer BRowCOOPtr = new Pointer();
		Pointer BRowCSRPtr = new Pointer();
		Pointer BColCSRPtr = new Pointer();
		Pointer BValCSRPtr = new Pointer();

		cudaMalloc(LuRowCOOPtr, nnzLu*Sizeof.INT);
		cudaMalloc(LuRowCSRPtr, (mLu+1)*Sizeof.INT);
		cudaMalloc(LuColCSRPtr, nnzLu*Sizeof.INT);
		cudaMalloc(LuValCSRPtr, nnzLu*Sizeof.DOUBLE);

		cudaMalloc(LbRowCOOPtr, nnzLb*Sizeof.INT);
		cudaMalloc(LbRowCSRPtr, (mLb+1)*Sizeof.INT);
		cudaMalloc(LbColCSRPtr, nnzLb*Sizeof.INT);
		cudaMalloc(LbValCSRPtr, nnzLb*Sizeof.DOUBLE);

		cudaMalloc(BRowCOOPtr, nnzB*Sizeof.INT);
		cudaMalloc(BRowCSRPtr, (mB+1)*Sizeof.INT);
		cudaMalloc(BColCSRPtr, nnzB*Sizeof.INT);
		cudaMalloc(BValCSRPtr, nnzB*Sizeof.DOUBLE);

		cudaMemcpy(LuRowCOOPtr, Pointer.to(LuRowCOO), nnzLu*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(LuColCSRPtr, Pointer.to(LuColCOO), nnzLu*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(LuValCSRPtr, Pointer.to(LuValCOO), nnzLu*Sizeof.DOUBLE, cudaMemcpyHostToDevice);

		cudaMemcpy(LbRowCOOPtr, Pointer.to(LbRowCOO), nnzLb*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(LbColCSRPtr, Pointer.to(LbColCOO), nnzLb*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(LbValCSRPtr, Pointer.to(LbValCOO), nnzLb*Sizeof.DOUBLE, cudaMemcpyHostToDevice);

		cudaMemcpy(BRowCOOPtr, Pointer.to(BRowCOO), nnzB*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(BColCSRPtr, Pointer.to(BColCOO), nnzB*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(BValCSRPtr, Pointer.to(BValCOO), nnzB*Sizeof.DOUBLE, cudaMemcpyHostToDevice);

		cusparseXcoo2csr(handle, LuRowCOOPtr, nnzLu, mLu, LuRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
		cusparseXcoo2csr(handle, LbRowCOOPtr, nnzLb, mLb, LbRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
		cusparseXcoo2csr(handle, BRowCOOPtr, nnzB, mB, BRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);

		cudaFree(LuRowCOOPtr);
		cudaFree(LbRowCOOPtr);
		cudaFree(BRowCOOPtr);



		m = mLb;
		n = labelCount;
		k = seedCount;
		int mRHS = m;
		int nRHS = n;
		Pointer RHSRowCSRPtr = new Pointer();
		Pointer RHSColCSRPtr = new Pointer();
		Pointer RHSValCSRPtr = new Pointer();
		Pointer nnzRHSPtr = new Pointer();

		int nnzRHSArr[] = new int[1];

		cudaMalloc(nnzRHSPtr, Sizeof.INT);
		cudaMalloc(RHSRowCSRPtr, (m+1)*Sizeof.INT);

		cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrLb, nnzLb, LbRowCSRPtr, LbColCSRPtr, descrB, nnzB, BRowCSRPtr, BColCSRPtr, descrRHS, RHSRowCSRPtr, nnzRHSPtr);

		cudaMemcpy(Pointer.to(nnzRHSArr), nnzRHSPtr, Sizeof.INT, cudaMemcpyDeviceToHost);

		int nnzRHS = nnzRHSArr[0];
		nnzRHSArr = null;

		cudaMalloc(RHSColCSRPtr, nnzRHS*Sizeof.INT);
		cudaMalloc(RHSValCSRPtr, nnzRHS*Sizeof.DOUBLE);

		cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrLb, nnzLb, LbValCSRPtr, LbRowCSRPtr, LbColCSRPtr, descrB, nnzB, BValCSRPtr, BRowCSRPtr, BColCSRPtr, descrRHS, RHSValCSRPtr, RHSRowCSRPtr, RHSColCSRPtr);

		Pointer RHSDensePtr = new Pointer();
		int ldRHS = mRHS;
		cudaMalloc(RHSDensePtr, mRHS*nRHS*Sizeof.DOUBLE);
		cusparseDcsr2dense(handle, mRHS, nRHS, descrRHS, RHSValCSRPtr, RHSRowCSRPtr, RHSColCSRPtr, RHSDensePtr, ldRHS);


		Pointer ProbPtr = new Pointer();
		cudaMalloc(ProbPtr, mLu*labelCount*Sizeof.DOUBLE);
		int ldProb = pixelCount-seedCount;
		double alpha[] = new double[] {1.0};
		cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo();
		cusparseCreateSolveAnalysisInfo(info);
		cusparseDcsrsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, pixelCount-seedCount, nnzLu, descrLu, LuValCSRPtr, LuRowCSRPtr, LuColCSRPtr, info);
		cusparseDcsrsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, pixelCount-seedCount, pixelCount-seedCount, Pointer.to(alpha), descrLu, LuValCSRPtr, LuRowCSRPtr, LuColCSRPtr, info, RHSDensePtr, ldRHS, ProbPtr, ldProb);
		
		double prob[] = new double[labelCount*mLu];
		double out[][] = new double[labelCount][mLu];
		cudaMemcpy(Pointer.to(prob), ProbPtr, mLu*pixelCount, cudaMemcpyDeviceToHost);
		System.out.printf("%d\n", prob.length);
	    for (int i = 0; i < labelCount; i++) { //print out array
	    	for (int j = 0; j < m; j++) {
	    		ix = i*m+j;
	    		out[i][j] = prob[ix];
	    		System.out.println(String.format("label no: %d, pixel no: %d, probability: %f", i, j, out[i][j]));
	    	}
	    }
	    return out;
		

	}
	
	public static <T extends Comparable<T>> void concurrentSort(
			final ArrayList<T> key, ArrayList<?>... lists){
		// Create a List of indices
		ArrayList<Integer> indices = new ArrayList<Integer>();
		for(int i = 0; i < key.size(); i++)
			indices.add(i);

		// Sort the indices list based on the key
		Collections.sort(indices, new Comparator<Integer>(){
			@Override public int compare(Integer i, Integer j) {
				return key.get(i).compareTo(key.get(j));
			}
		});

		// Create a mapping that allows sorting of the List by N swaps.
		// Only swaps can be used since we do not know the type of the lists
		Map<Integer,Integer> swapMap = new HashMap<Integer, Integer>(indices.size());
		ArrayList<Integer> swapFrom = new ArrayList<Integer>(indices.size()),
				swapTo   = new ArrayList<Integer>(indices.size());
		for(int i = 0; i < key.size(); i++){
			int k = indices.get(i);
			while(i != k && swapMap.containsKey(k))
				k = swapMap.get(k);

			swapFrom.add(i);
			swapTo.add(k);
			swapMap.put(i, k);
		}

		// use the swap order to sort each list by swapping elements
		for(ArrayList<?> list : lists)
			for(int i = 0; i < list.size(); i++)
				Collections.swap(list, swapFrom.get(i), swapTo.get(i));
	}
	
	private static double weight(int a, int b, double beta) {

		Color color1 = new Color(a);
		Color color2 = new Color(b);

		int red = (int) Math.pow(color1.getRed() - color2.getRed(), 2);
		int green = (int) Math.pow(color1.getGreen() - color2.getGreen(), 2);
		int blue = (int) Math.pow(color1.getBlue() - color2.getBlue(), 2);
		int alpha = (int) Math.pow(color1.getAlpha() - color2.getAlpha(), 2);
		int sum = red+green+blue+alpha;

		return (double) Math.exp(-(beta*sum));

	}
	
	private static int numUnique(int[] arr) { //gets number of unique elements in given array
		ArrayList<Integer> unique = new ArrayList<Integer>();
		int cur;
		for (int i = 0; i < arr.length; i++) {
			cur = arr[i];
			if (!unique.contains(cur)) {
				unique.add(cur);
			}
		}
		
		return unique.size();
		
	}
}
