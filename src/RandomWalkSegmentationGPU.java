import java.awt.Color;
import java.util.ArrayList;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparseSolveAnalysisInfo;
import jcuda.runtime.JCuda;
import jcuda.jcusparse.cusparseStatus;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;

public class RandomWalkSegmentationGPU {
	static double[][] getProbabilities(int[] pixels, int pixelCount, Edge[] edges, int edgeCount, double beta, int[] seeds, int[] labels) {
		
		/*
		 * Some initialisation stuff, as recommended in jcuda tutorials
		 */
		
	    //JCusparse.setExceptionsEnabled(true); //enable cusparse stuff
	    //JCuda.setExceptionsEnabled(true);
	    
	    cusparseHandle handle = new cusparseHandle(); //create handle 
	    cusparseCreate(handle);
	    
		
		/*
		 * Create matrix C in CSR format (first populate in COO and then convert)
		 * read in row-major (left to right, top to bottom)
		 * diagonal matrix of edge weights
		 * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-coo2csr
		 */
	    
		int nnzC = edgeCount; //number of nonzero elements in constitutive matrix
		//matrix C creation
	    int CRowCoo[] = new int[nnzC]; //arrays to hold matrix C in COO format
	    int CColCoo[] = new int[nnzC];
	    double CValCoo[] = new double[nnzC];
	    double totweight = 0;
	    for (int i = 0; i < nnzC; i++) { //populate COO representation of C
	    	CRowCoo[i] = i;//diagonal matrix of edge weights
	    	CColCoo[i] = i;
	    	totweight += weight(pixels[edges[i].start], pixels[edges[i].end], beta);
	    	CValCoo[i] = weight(pixels[edges[i].start], pixels[edges[i].end], beta);
	    }
	    double avgweight = totweight/nnzC;
	    
	    Pointer CRowCooPtr = new Pointer(); //pointers for creating CSR matrix C
	    Pointer CRowCSRPtr = new Pointer();
	    Pointer CColCSRPtr = new Pointer();
	    Pointer CValCSRPtr = new Pointer();
	    
	    cudaMalloc(CRowCooPtr, nnzC*Sizeof.INT); //allocate memory for pointers
	    cudaMalloc(CRowCSRPtr, (edgeCount+1)*Sizeof.INT); //1+num rows
	    cudaMalloc(CColCSRPtr, nnzC*Sizeof.INT);
	    cudaMalloc(CValCSRPtr, nnzC*Sizeof.DOUBLE);
	    
	    //copy values into pointers
	    cudaMemcpy(CRowCooPtr, Pointer.to(CRowCoo), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(CColCSRPtr, Pointer.to(CColCoo), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(CValCSRPtr, Pointer.to(CValCoo), nnzC*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    //convert row to CSR representation
	    cusparseXcoo2csr(handle, CRowCooPtr, nnzC, nnzC, CRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    JCuda.cudaDeviceSynchronize();
	    
	    /*
	     * Printing stuff for testing!!
	     */
	    //int printRow[] = new int[nnzC];
	    //int printCol[] = new int[nnzC];
	    //double printVal[] = new double[nnzC];
	    
	   // cudaMemcpy(Pointer.to(printRow), CRowCooPtr, nnzC*Sizeof.INT, cudaMemcpyDeviceToHost);
	    //cudaMemcpy(Pointer.to(printCol), CColCSRPtr, nnzC*Sizeof.INT, cudaMemcpyDeviceToHost);
	    //cudaMemcpy(Pointer.to(printVal), CValCSRPtr, nnzC*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix C: %d entries\n", nnzC);
	    
	    //for (int i = 0; i < nnzC; i++) {
	    	//System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    //}
	    /*
	     * Printing stuff for testing!!
	     */

	    //free up coo representation of row
	    cudaFree(CRowCooPtr);
		
	    /*
	     * Create matrix A in CSR format (first populate in COO and then convert)
	     * read in row-major (left to right, top to bottom)
	     * edge*pixel matrix, where each row has an entry at the pixel corresponding to
	     * each end of the edge. (-1 for one end and 1 for the other)
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-coo2csr
	     */
	    
	    int nnzA = edgeCount*2; //number of nonzero elements in incidence matrix
	    //Matrix A creation
		int ARowCoo[] = new int[nnzA]; //arrays to hold matrix A in COO format
	    int AColCoo[] = new int[nnzA];
	    double AValCoo[] = new double[nnzA];
	    //populate matrix A
	    int AEntries = 0;
	    
	    for (int i = 0; i < edgeCount; i++) { //for each edge
	    	ARowCoo[AEntries] = i;
	    	AColCoo[AEntries] = edges[i].start;
	    	AValCoo[AEntries++] = -1; //assign edge an orientation -1 in source, +1 in dest
	    	ARowCoo[AEntries] = i;
	    	AColCoo[AEntries] = edges[i].end;
	    	AValCoo[AEntries++] = 1;
	    }
	    
	    //pointers for csr representation for A
	    Pointer ARowCooPtr = new Pointer();
	    Pointer ARowCSRPtr = new Pointer();
	    Pointer AColCSRPtr = new Pointer();
	    Pointer AValCSRPtr = new Pointer();
	    
	    //allocate memory for pointers
	    cudaMalloc(ARowCooPtr, nnzA*Sizeof.INT);
	    cudaMalloc(ARowCSRPtr, (edgeCount+1)*Sizeof.INT); //1+num rows
	    cudaMalloc(AColCSRPtr, nnzA*Sizeof.INT);
	    cudaMalloc(AValCSRPtr, nnzA*Sizeof.DOUBLE);
	    
	    //copy values into pointers
	    cudaMemcpy(ARowCooPtr, Pointer.to(ARowCoo), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(AColCSRPtr, Pointer.to(AColCoo), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(AValCSRPtr, Pointer.to(AValCoo), nnzA*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    //convert row representation to CSR
	    cusparseXcoo2csr(handle, ARowCooPtr, nnzA, edgeCount, ARowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    JCuda.cudaDeviceSynchronize();
	    
	    /*
	     * Printing stuff for testing!!
	     */
	    //printRow = new int[nnzA];
	    //printCol = new int[nnzA];
	    //printVal = new double[nnzA];
	    
	    //cudaMemcpy(Pointer.to(printRow), ARowCooPtr, nnzA*Sizeof.INT, cudaMemcpyDeviceToHost);
	    //cudaMemcpy(Pointer.to(printCol), AColCSRPtr, nnzA*Sizeof.INT, cudaMemcpyDeviceToHost);
	    //cudaMemcpy(Pointer.to(printVal), AValCSRPtr, nnzA*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix A: %d entries\n", nnzA);
	    
	    //for (int i = 0; i < nnzA; i++) {
	    	//System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    //}
	    /*
	     * Printing stuff for testing!!
	     */
	    
	    
	    //free up row coo memory

		cudaFree(ARowCooPtr);
		
		/*
		 * Create and set up matrix descriptions to calculate laplacian
		 */
	    
	    //declare matrix descriptions
	    cusparseMatDescr descrA = new cusparseMatDescr();
	    cusparseMatDescr descrC = new cusparseMatDescr();
	    cusparseMatDescr descrAtC = new cusparseMatDescr();
	    cusparseMatDescr descrLap = new cusparseMatDescr();
	    
	    //create the matrix description
	    cusparseCreateMatDescr(descrA);
	    cusparseCreateMatDescr(descrC);
	    cusparseCreateMatDescr(descrAtC);
	    cusparseCreateMatDescr(descrLap);
	    
	    //set the matrix type
	    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrAtC, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrLap, CUSPARSE_MATRIX_TYPE_GENERAL);
	    
	    //set the matrix index
	    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrAtC, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrLap, CUSPARSE_INDEX_BASE_ZERO);
		
	    
		//Lap = A^T*C*A
		
		
	    /*
	     * Perform first half of laplacian calculation (A^T*C)
	     * use csrgemm method to perform sparse matrix multiplication
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm
	     */
	    
		//first half of lap calculation (A^T*C)
	    int m,n,k; //A^T is mxk matrix, C is kxn  A^T*C is mxn
	    //set m n and k
	    m = pixelCount; 
	    n = edgeCount;
	    k = edgeCount;
	    
	    //pointers for intermediate result
	    Pointer AtCRowCSRPtr = new Pointer();
	    Pointer AtCColCSRPtr = new Pointer();
	    Pointer AtCValCSRPtr = new Pointer();
	    Pointer nnzAtCPtr = new Pointer();
	    
	    //num non zero elements for intermediate result
	    int nnzAtC[] = new int[1];
	    
	    //allocate memory for row and nnz pointers
	    cudaMalloc(nnzAtCPtr, Sizeof.INT);
	    cudaMalloc(AtCRowCSRPtr, (m+1)*Sizeof.INT);
	    
	    //find nnz for result
	    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA, ARowCSRPtr, AColCSRPtr, descrC, nnzC, CRowCSRPtr, CColCSRPtr, descrAtC, AtCRowCSRPtr, nnzAtCPtr);
	    JCuda.cudaDeviceSynchronize();

	    
	    //copy nnz back to host memory
	    cudaMemcpy(Pointer.to(nnzAtC), nnzAtCPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("nnzAtC: %d\n", nnzAtC[0]);
	    
	    cudaFree(nnzAtCPtr);
	    
	    //allocate memory for intermediate result pointers
	    cudaMalloc(AtCColCSRPtr, nnzAtC[0]*Sizeof.INT);
	    cudaMalloc(AtCValCSRPtr, nnzAtC[0]*Sizeof.DOUBLE);
	    
	    //perform multiplication
	    cusparseDcsrgemm(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA, AValCSRPtr, ARowCSRPtr, AColCSRPtr, descrC, nnzC, CValCSRPtr, CRowCSRPtr, CColCSRPtr, descrAtC, AtCValCSRPtr, AtCRowCSRPtr, AtCColCSRPtr);
	    JCuda.cudaDeviceSynchronize();
	    
	    //free CRowCSRPtr, CColCSRPtr, CValCSRPtr
	    cudaFree(CRowCSRPtr);
	    cudaFree(CColCSRPtr);
	    cudaFree(CValCSRPtr);
	    
	    /*
	     * Printing stuff for testing!!
	     *
	    printRow = new int[nnzAtC[0]];
	    printCol = new int[nnzAtC[0]];
	    printVal = new double[nnzAtC[0]];
	    
	    Pointer AtCRowCooPtr = new Pointer();
	    cudaMalloc(AtCRowCooPtr, nnzAtC[0]*Sizeof.INT);
	    cusparseXcsr2coo(handle, AtCRowCSRPtr, nnzAtC[0], m, AtCRowCooPtr, CUSPARSE_INDEX_BASE_ZERO);
	    cudaMemcpy(Pointer.to(printRow), AtCRowCooPtr, nnzAtC[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printCol), AtCColCSRPtr, nnzAtC[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printVal), AtCValCSRPtr, nnzAtC[0]*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix AtC: %d entries\n", nnzAtC[0]);
	    
	    for (int i = 0; i < nnzAtC[0]; i++) {
	    	//System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    }
	    /*
	     * Printing stuff for testing!!
	     */
	    
	    //System.out.println("A^T*C calculated");
	    
	    /*
	     * Perform second half of laplacian calculation (A^T*C*A)
	     * use csrgemm method for sparse matrix multiplication
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm
	     */
	    
	    //second half of laplacian calculation
	    Pointer LapRowCSRPtr = new Pointer();
	    Pointer LapColCSRPtr = new Pointer();
	    Pointer LapValCSRPtr = new Pointer();
	    Pointer nnzLapPtr = new Pointer();
	    
	    //num non zero elements in lap
	    int nnzLap[] = new int[1];
	    
	    //m, n, k for lap, AtC and A
	    m = pixelCount;
	    n = pixelCount;
	    k = edgeCount;

	    //allocate memory for nnz and row 
	    cudaMalloc(nnzLapPtr, Sizeof.INT);
	    cudaMalloc(LapRowCSRPtr, (m+1)*Sizeof.INT);
	    
	    //find nnz for lap
	    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrAtC, nnzAtC[0], AtCRowCSRPtr, AtCColCSRPtr, descrA, nnzA, ARowCSRPtr, AColCSRPtr, descrLap, LapRowCSRPtr, nnzLapPtr);
	    JCuda.cudaDeviceSynchronize();
	    
	    //copy nnz lap back to host mem
	    cudaMemcpy(Pointer.to(nnzLap), nnzLapPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("nnzLap : %d\n", nnzLap[0]);
	    
	    //free nnzLapPtr
	    cudaFree(nnzLapPtr);
	    
	    //allocate memory for col and vals
	    cudaMalloc(LapColCSRPtr, nnzLap[0]*Sizeof.INT);
	    cudaMalloc(LapValCSRPtr, nnzLap[0]*Sizeof.DOUBLE);
	    
	    //perform multiplication to get laplacian
	    cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrAtC, nnzAtC[0], AtCValCSRPtr, AtCRowCSRPtr, AtCColCSRPtr, descrA, nnzA, AValCSRPtr, ARowCSRPtr, AColCSRPtr, descrLap, LapValCSRPtr, LapRowCSRPtr, LapColCSRPtr);
	    JCuda.cudaDeviceSynchronize();
	    
	    //free AtCRowCSRPtr, AtCColCSRPtr, AtCValCSRPtr, ARowCSRPtr, AColCSRPtr, AValCSRPtr
	    cudaFree(AtCRowCSRPtr);
	    cudaFree(AtCColCSRPtr);
	    cudaFree(AtCValCSRPtr);
	    cudaFree(ARowCSRPtr);
	    cudaFree(AColCSRPtr);
	    cudaFree(AValCSRPtr);
	    
	    /*
	     * Printing stuff for testing!!
	     *
	    printRow = new int[nnzLap[0]];
	    printCol = new int[nnzLap[0]];
	    printVal = new double[nnzLap[0]];
	    
	    Pointer LapRowCooPtr1 = new Pointer();
	    cudaMalloc(LapRowCooPtr1, nnzLap[0]*Sizeof.INT);
	    cusparseXcsr2coo(handle, LapRowCSRPtr, nnzLap[0], m, LapRowCooPtr1, CUSPARSE_INDEX_BASE_ZERO);
	    cudaMemcpy(Pointer.to(printRow), LapRowCooPtr1, nnzLap[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printCol), LapColCSRPtr, nnzLap[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printVal), LapValCSRPtr, nnzLap[0]*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix Lap: %d entries\n", nnzLap[0]);
	    
	    for (int i = 0; i < nnzLap[0]; i++) {
	    	//System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    }
	    /*
	     * Printing stuff for testing!!
	     */
	    
	   // System.out.println("Laplacian calculated");
	    
	    /*
	     * To figure out the RHS of the equation, and the LHS (just unseeded part of laplacian)
	     * LHS is unseeded part of laplacian, and RHS is formed using
	     * a part of the laplacian as well
	     * in order to get a part of the laplacian, we convert it back to COO
	     * Then remove the parts we need to remove
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csr2coo
	     */
	    
	    int num_labels = numUnique(labels); //number of unique labels
	    int num_seeds = seeds.length; //number of seeds
	    
	    int[] LapColCoo = new int[nnzLap[0]]; //convert lap back to COO 
	    int[] LapRowCoo = new int[nnzLap[0]]; //needed to find matrices for forming linear system
	    double[] LapValCoo = new double[nnzLap[0]];
	    
	    Pointer LapRowCooPtr = new Pointer(); //pointer for lap row co
	    
	    cudaMalloc(LapRowCooPtr, nnzLap[0]*Sizeof.INT); //alloc mem
	    
	    //convert csr back to coo
	    cusparseXcsr2coo(handle, LapRowCSRPtr, nnzLap[0], m, LapRowCooPtr, CUSPARSE_INDEX_BASE_ZERO);
	    JCuda.cudaDeviceSynchronize();
	    
	    //copy values into coo arrays
	    cudaMemcpy(Pointer.to(LapColCoo), LapColCSRPtr, nnzLap[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(LapRowCoo), LapRowCooPtr, nnzLap[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(LapValCoo), LapValCSRPtr, nnzLap[0]*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    	    
	    //System.out.println("Laplacian converted back to COO");
	    //free: LapRowCooPtr, LapRowCSRPtr, LapColCSRPtr, LapValCSRPtr
	    cudaFree(LapRowCooPtr);
	    cudaFree(LapRowCSRPtr);
	    cudaFree(LapColCSRPtr);
	    cudaFree(LapValCSRPtr);
	    
	    /*
	     * To get Lu (LHS of equation), we remove all the seed pixel rows/columns
	     * 
	     */
	    
	    //coo representation for Lu (LHS of eq)
	    ArrayList<Integer> LuRowCoo = new ArrayList<Integer>();
	    ArrayList<Integer> LuColCoo = new ArrayList<Integer>();
	    ArrayList<Double> LuValCoo = new ArrayList<Double>();
	    
	    ArrayList<Integer> LbRowCoo = new ArrayList<Integer>(); //coo for Lb (part of RHS of eq)
	    ArrayList<Integer> LbColCoo = new ArrayList<Integer>();
	    ArrayList<Double> LbValCoo = new ArrayList<Double>();
	    //System.out.println("ArrayLists created");
	    
	    for (int i = 0; i < nnzLap[0]; i++) { //populate array lists
	    	LuRowCoo.add(LapRowCoo[i]); //pls give better solution
	    	LuColCoo.add(LapColCoo[i]); //I cant find a way to directly
	    	LuValCoo.add(LapValCoo[i]); //make an arraylist from an array
	    	LbRowCoo.add(LapRowCoo[i]);
	    	LbColCoo.add(LapColCoo[i]);
	    	LbValCoo.add(LapValCoo[i]);	    	
	    	//if (LapValCoo[i] != (double) 0) System.out.printf("Row: %d, Col: %d, Val: %f\n", LapRowCoo[i], LapColCoo[i], LapValCoo[i]);
	    }
	   // System.out.println("ArrayLists populated");
	     
	    int ix; //index
	    int nnzLu = nnzLap[0]; //num nonzero for Lu and Lb
	    int nnzLb = nnzLap[0];
	    
	    int cur;
	    int seeds_temp[] = seeds.clone(); //get copy of seeds (we modify during these methods)
	    
	    for (int i = 0; i < num_seeds; i++) { //for each seed
	    	ix = LuRowCoo.indexOf(seeds_temp[i]);
	    	while (ix != -1) { //remove seed rows
	    		LuRowCoo.remove(ix);
	    		LuColCoo.remove(ix);
	    		LuValCoo.remove(ix);
	    		nnzLu--;
	    		ix = LuRowCoo.indexOf(seeds_temp[i]);
	    	}
	    	for (int j = 0; j < LuRowCoo.size(); j++) { //subtract 1 from every row index above it (as it has been removed)
	    		cur = LuRowCoo.get(j);
	    		if (cur > seeds_temp[i]) {
	    			LuRowCoo.set(j, cur-1);
	    		}
	    	}
	    	for (int j = 0; j < seeds.length; j++) { //subtract 1 from every seed above it (corresponds to row movement)
	    		if (seeds_temp[j]>seeds_temp[i]) {
	    			seeds_temp[j]--;
	    		}
	    	}
	    }
	    seeds_temp = seeds.clone();
	    for (int i = 0; i < num_seeds; i++) { //for each seed
	    	ix = LuColCoo.indexOf(seeds_temp[i]);
	    	while (ix != -1) { //remove seed cols
	    		LuRowCoo.remove(ix);
	    		LuColCoo.remove(ix);
	    		LuValCoo.remove(ix);
	    		nnzLu--;
	    		ix = LuColCoo.indexOf(seeds_temp[i]);
	    	}
	    	for (int j = 0; j < LuColCoo.size(); j++)  { //subtract 1 from every column index above it (as the row beneath has been removed)
	    		cur = LuColCoo.get(j);
	    		if (cur > seeds_temp[i]) {
	    			LuColCoo.set(j, cur-1);
	    		}
	    	}
	    	for (int j = 0; j < seeds.length; j++) { //subtract 1 from every seed above it (corresponds to column movement)
	    		if (seeds_temp[j] > seeds_temp[i]) {
	    			seeds_temp[j]--;
	    		}
	    	}
	    }
	    
	    int LuRowCooArr[] = new int[nnzLu]; //put back in to proper primitive types
	    int LuColCooArr[] = new int[nnzLu];
	    double LuValCooArr[] = new double[nnzLu];
	    
	    for (int i = 0; i < nnzLu; i++) {
	    	LuRowCooArr[i] = LuRowCoo.get(i);
	    	LuColCooArr[i] = LuColCoo.get(i);
	    	LuValCooArr[i] = LuValCoo.get(i);
	    }
	    
	    /*
	     * To get Lb (part of RHS), we remove seed rows and unseeded columns from the
	     * laplacian
	     */
	    
	    ArrayList<Integer> seed = new ArrayList<Integer>(); //get seeds in arraylist
	    for (int i = 0; i < num_seeds; i++) { //get seeds in arraylist
	    	seed.add(seeds[i]); //just so we get .contains method
	    }
	    
	    seeds_temp = seeds.clone();
	    for (int i = 0; i < num_seeds; i++) {
	    	ix = LbRowCoo.indexOf(seeds_temp[i]);
	    	while (ix != -1) { //remove seed rows
	    		LbRowCoo.remove(ix);
	    		LbColCoo.remove(ix);
	    		LbValCoo.remove(ix);
	    		nnzLb--;
	    		ix = LbRowCoo.indexOf(seeds_temp[i]);
	    	}
	    	for (int j = 0; j < LbRowCoo.size(); j++) { //subtract 1 from every row index above it (as it has been removed)
	    		cur = LbRowCoo.get(j);
	    		if (cur > seeds_temp[i]) {
	    			LbRowCoo.set(j, cur-1);
	    		}
	    	}
	    	for (int j = 0; j < seeds.length; j++) { //subtract 1 from every seed above it (corresponds to row movement)
	    		if (seeds_temp[j]>seeds_temp[i]) {
	    			seeds_temp[j]--;
	    		}
	    	}
	    }
	    int unseeded[] = new int[pixelCount-num_seeds];
	    int index = 0;
	    for (int i = 0; i < pixelCount; i++) { //populate array of unseeded pixels
	    	if (!seed.contains(i)) unseeded[index++] = i;
	    }
	    for (int i = 0; i < pixelCount-num_seeds; i++) { //remove unseeded columns
	    	ix = LbColCoo.indexOf(unseeded[i]);
	    	while (ix != -1) {
	    		LbRowCoo.remove(ix);
	    		LbColCoo.remove(ix);
	    		LbValCoo.remove(ix);
	    		nnzLb--;
	    		ix = LbColCoo.indexOf(unseeded[i]);
	    	}
	    	for (int j = 0; j < LbColCoo.size(); j++) { //shuffle down column indices
	    		cur = LbColCoo.get(j);
	    		if (cur > unseeded[i]) {
	    			LbColCoo.set(j, cur-1);
	    		}
	    	}
	    	for (int j = 0; j < unseeded.length; j++) { //shuffle down unseeded array
	    		if (unseeded[j] > unseeded[i]) {
	    			unseeded[j]--;
	    		}
	    	}
	    }
	   
	    
	    /*
	     * create matrix b (second half of RHS)
	     * RHS is -Lb*b
	     * b is seed*labels matrix, with a 1 in row,column if seed[row] is label[column]
	     */
	    
	    m = num_seeds; //num rows of b
	    n = num_labels; //num cols of b

	    int nnzB = num_seeds; //num non zero elements
	    
	    int bRowCoo[] = new int[nnzB]; //arrays to hold b in coo format
	    int bColCoo[] = new int[nnzB];
	    double bValCoo[] = new double[nnzB];
	    
	    for (int i = 0; i < num_seeds; i++) { //set up b
	    	bRowCoo[i] = i;
	    	bColCoo[i] = labels[i];
	    	bValCoo[i] = 1;
	    }
	    
	    /*
	     * Perform multiplication -Lb*b on GPU
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm
	     */
	   
	    int LbRowCooArr[] = new int[nnzLb];
	    int LbColCooArr[] = new int[nnzLb];
	    double LbValCooArr[] = new double[nnzLb];
	    for (int i = 0; i < nnzLb; i++) {
	    	LbRowCooArr[i] = LbRowCoo.get(i); 
	    	LbColCooArr[i] = LbColCoo.get(i);
	    	LbValCooArr[i] = -LbValCoo.get(i);//value is neg because we want -Lb*b
	    }
	    
	    /*
	     * convert matrix b to Csr representation and allocate on gpu
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-coo2csr
	     */
	    
	    //allocate matrix b on GPU
	    Pointer bRowCooPtr = new Pointer();//pointers
	    Pointer bRowCSRPtr = new Pointer();
	    Pointer bColCSRPtr = new Pointer();
	    Pointer bValCSRPtr = new Pointer();
	    
	    
	    cudaMalloc(bRowCooPtr, nnzB*Sizeof.INT); //allocate memory for pointers
	    cudaMalloc(bRowCSRPtr, (num_seeds+1)*Sizeof.INT);
	    cudaMalloc(bColCSRPtr, nnzB*Sizeof.INT);
	    cudaMalloc(bValCSRPtr, nnzB*Sizeof.DOUBLE);
	    
	    cudaMemcpy(bRowCooPtr, Pointer.to(bRowCoo), nnzB*Sizeof.INT, cudaMemcpyHostToDevice);//copy values in to pointers
	    cudaMemcpy(bColCSRPtr, Pointer.to(bColCoo), nnzB*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(bValCSRPtr, Pointer.to(bValCoo), nnzB*Sizeof.DOUBLE, cudaMemcpyHostToDevice);	    
	    
	    cusparseXcoo2csr(handle, bRowCooPtr, nnzB, nnzB, bRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO); //convert COO row representation to CSR (for multiplication later)
	    JCuda.cudaDeviceSynchronize(); //sync up
	    
	    /*
	     * Printing stuff for testing!!
	     *
	    printRow = new int[nnzB];
	    int[] printRowCSR = new int[3];
	    printCol = new int[nnzB];
	    printVal = new double[nnzB];
	    
	    cudaMemcpy(Pointer.to(printRowCSR), bRowCSRPtr, (3)*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printRow), bRowCooPtr, nnzB*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printCol), bColCSRPtr, nnzB*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printVal), bValCSRPtr, nnzB*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix b: %d entries\n", nnzB);
	    
	    for (int i = 0; i < nnzB; i++) {
	    	//System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    }
	    for (int i = 0; i < 3; i++) {
	    	//System.out.print(printRowCSR[i]);
	    	//System.out.print(',');
	    }
	    //System.out.print('\n');
	    /*
	     * Printing stuff for testing!!
	     */
	    
	    
	    cudaFree(bRowCooPtr);//free up unneeded memory

	    
	    //System.out.println("b calculated and assigned on gpu");
	    
	    /*
	     * convert matrix Lb to CSR representation and allocate on GPU
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-coo2csr
	     */
	    m = pixelCount - num_seeds;
	    n = num_seeds;
	    
	    Pointer LbRowCooPtr = new Pointer(); //form matrix lb
	    Pointer LbRowCSRPtr = new Pointer(); //assign on gpu
	    Pointer LbColCSRPtr = new Pointer(); //create pointers
	    Pointer LbValCSRPtr = new Pointer();
	    
	    cudaMalloc(LbRowCooPtr, nnzLb*Sizeof.INT); //allocate memory for pointers
	    cudaMalloc(LbRowCSRPtr, (m+1)*Sizeof.INT);
	    cudaMalloc(LbColCSRPtr, nnzLb*Sizeof.INT);
	    cudaMalloc(LbValCSRPtr, nnzLb*Sizeof.DOUBLE);
	    
	    cudaMemcpy(LbRowCooPtr, Pointer.to(LbRowCooArr), nnzLb*Sizeof.INT, cudaMemcpyHostToDevice); //copy values into pointer
	    cudaMemcpy(LbColCSRPtr, Pointer.to(LbColCooArr), nnzLb*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(LbValCSRPtr, Pointer.to(LbValCooArr), nnzLb*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    //System.out.println("Lb Formation");
	    //System.out.println(MatrixUtils.PointerContents(LbRowCooPtr, nnzLb, true));
	    //System.out.println(MatrixUtils.PointerContents(LbColCSRPtr, nnzLb, true));
	    //System.out.println(MatrixUtils.PointerContents(LbValCSRPtr, nnzLb, false));
	    
	    cusparseXcoo2csr(handle, LbRowCooPtr, nnzLb, m, LbRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO); //convert COO representation into csr representation
	    //System.out.println(MatrixUtils.PointerContents(LbRowCSRPtr, m+1, true));
	    JCuda.cudaDeviceSynchronize(); //synchronise
	    
	    /*
	     * Printing stuff for testing!!
	     *
	    printRow = new int[nnzLb];
	    printCol = new int[nnzLb];
	    printVal = new double[nnzLb];
	    
	    cudaMemcpy(Pointer.to(printRow), LbRowCooPtr, nnzLb*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printCol), LbColCSRPtr, nnzLb*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printVal), LbValCSRPtr, nnzLb*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix Lb: %d entries\n", nnzLb);
	    
	    for (int i = 0; i < nnzLb; i++) {
	    	//System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    }
	    /*
	     * Printing stuff for testing!!
	     */
	    
	    
	    //cudaFree(LbRowCooPtr);	    
	    
	    //System.out.println("Lb calculated and assigned on gpu");
	    
	    /*
	     * create pointers and matrix descriptions for -Lb*b multiplication
	     * 
	     */
	    
	    cusparseMatDescr descrb = new cusparseMatDescr(); //format descriptions for matrices
	    cusparseMatDescr descrLb = new cusparseMatDescr();
	    cusparseMatDescr descrRHS = new cusparseMatDescr();
	    
	    cusparseCreateMatDescr(descrb); //create matrix descriptions
	    cusparseCreateMatDescr(descrLb);
	    cusparseCreateMatDescr(descrRHS);
	    
	    cusparseSetMatType(descrb, CUSPARSE_MATRIX_TYPE_GENERAL); //set matrix description types
	    cusparseSetMatType(descrLb, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrRHS, CUSPARSE_MATRIX_TYPE_GENERAL);
	    
	    cusparseSetMatIndexBase(descrb, CUSPARSE_INDEX_BASE_ZERO); //set index base
	    cusparseSetMatIndexBase(descrLb, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrRHS, CUSPARSE_INDEX_BASE_ZERO);
	    
	    /*
	     * Perform multiplication Lb*b
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm
	     */
	    
	    Pointer RHSRowCSRPtr = new Pointer(); //right hand side of problem
	    Pointer RHSColCSRPtr = new Pointer(); // is -lb*b
	    Pointer RHSValCSRPtr = new Pointer(); //pointers for right hand side as sparse matrix (use sparse matrix multipication)
	    Pointer nnzRHSPtr = new Pointer();
	    
	    int nnzRHS[] = new int[1];
	    m = pixelCount - num_seeds; //m,n,k for multiplication -Lb*b
	    k = num_seeds; //Lb is m*k matrix
	    n = num_labels; //b is k*n matrix	    
	    
	    cudaMalloc(RHSRowCSRPtr, (m+1)*Sizeof.INT); //create RHS
	    cudaMalloc(nnzRHSPtr, Sizeof.INT); //allocate memory for rhs row
	    
	    //just seems to be this line that is the problem
	    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrLb, nnzLb, LbRowCSRPtr, LbColCSRPtr, descrb, nnzB, bRowCSRPtr, bColCSRPtr, descrRHS, RHSRowCSRPtr, nnzRHSPtr);
	    JCuda.cudaDeviceSynchronize(); //find nnz for rhs, synchronise
	    
	    //or this doesnt function correctly
	    cudaMemcpy(Pointer.to(nnzRHS), nnzRHSPtr, Sizeof.INT, cudaMemcpyDeviceToHost); //copy nnz back to host
	    //cudaFree(nnzRHSPtr);
	    
	    cudaMalloc(RHSColCSRPtr, nnzRHS[0]*Sizeof.INT); //allocate memory for col and value pointers
	    cudaMalloc(RHSValCSRPtr, nnzRHS[0]*Sizeof.DOUBLE);
	    cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrLb, nnzLb, LbValCSRPtr, LbRowCSRPtr, LbColCSRPtr, descrb, nnzB, bValCSRPtr, bRowCSRPtr, bColCSRPtr, descrRHS, RHSValCSRPtr, RHSRowCSRPtr, RHSColCSRPtr);
	    JCuda.cudaDeviceSynchronize(); //perform multiplication and synchronise
	    
	    cudaFree(bRowCSRPtr); //free up pointers that are no-longer needed
	    cudaFree(bColCSRPtr);
	    cudaFree(bValCSRPtr);
	    cudaFree(LbRowCSRPtr);
	    cudaFree(LbColCSRPtr);
	    cudaFree(LbValCSRPtr);
	    
	    /*
	     * print out CSR representation to check if issue is with conversion (its not)
	     */
	    
	    int print[] = new int[m+1];
	    cudaMemcpy(Pointer.to(print), RHSRowCSRPtr, (m+1)*Sizeof.INT, cudaMemcpyDeviceToHost);
	    //for (int i = 0; i < m+1; i++) {
	    	//System.out.println(print[i]);
	    //}
	    	    
	    /*
	     * Printing stuff for testing!!
	     *
	    printRow = new int[nnzRHS[0]];
	    printCol = new int[nnzRHS[0]];
	    printVal = new double[nnzRHS[0]];
	    
	    Pointer RHSRowCooPtr = new Pointer();
	    cudaMalloc(RHSRowCooPtr, nnzRHS[0]*Sizeof.INT);
	    cusparseXcsr2coo(handle, RHSRowCSRPtr, nnzRHS[0], m, RHSRowCooPtr, CUSPARSE_INDEX_BASE_ZERO);
	    
	    cudaMemcpy(Pointer.to(printRow), RHSRowCooPtr, nnzRHS[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printCol), RHSColCSRPtr, nnzRHS[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printVal), RHSValCSRPtr, nnzRHS[0]*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    System.out.printf("Matrix RHS(sparse): %d entries\n", nnzRHS[0]);
	    
	    for (int i = 0; i < nnzRHS[0]; i++) {
	    	System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    }
	    /*
	     * Printing stuff for testing!!
	     */
	    
	    
	    //System.out.println("RHS calculated and assigned on gpu");
	    
	    //eq will be Lu*X=-Lb*Bound where bound is a seedcount*labelcount matrix, where an entry is 1 if that seed (row) has the same label as the col
	    
	    /*
	     * Convert RHS matrix to a dense matrix, because solver requires it
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csr2dense
	     */
	    
	    //System.out.println("RHS nnz");
	    //System.out.println(nnzRHS[0]);
	    
	    
	    Pointer RHSRowCooPtr = new Pointer();
	    cudaMalloc(RHSRowCooPtr, nnzRHS[0]*Sizeof.INT);
	    cusparseXcsr2coo(handle, RHSRowCSRPtr, nnzRHS[0], m, RHSRowCooPtr, CUSPARSE_INDEX_BASE_ZERO);
	    
	    int[] RHSRow = new int[nnzRHS[0]];
	    int[] RHSCol = new int[nnzRHS[0]];
	    double[] RHSVal = new double[nnzRHS[0]];
	    
	    cudaMemcpy(Pointer.to(RHSRow), RHSRowCooPtr, nnzRHS[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(RHSCol), RHSColCSRPtr, nnzRHS[0]*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(RHSVal), RHSValCSRPtr, nnzRHS[0]*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.println(MatrixUtils.PointerContents(RHSRowCooPtr, nnzRHS[0], true));
	    //System.out.println(MatrixUtils.PointerContents(RHSColCSRPtr, nnzRHS[0], true));
	    //System.out.println(MatrixUtils.PointerContents(RHSValCSRPtr, nnzRHS[0], false));
	    
	    //System.out.println("RHS");
	    
	    //for (int i = 0; i < nnzRHS[0]; i++) {
	    	//System.out.print(RHSRow[i]);
	    	//System.out.print(" ");
	    	//System.out.print(RHSCol[i]);
	    	//System.out.print(" ");
	    	//System.out.print(RHSVal[i]);
	    	//System.out.print("\n");	    	
	    //}
	    
	    //System.out.print(m);
	    //System.out.print(" ");
	    //System.out.print(n);
	    //System.out.print("\n");
	    
	    //double[][] data = new double[m][n];
	    
	    //for (int i = 0; i < nnzRHS[0]; i++) {
	    	//data[RHSRow[i]][RHSCol[i]] = RHSVal[i];
	    //}
	    
	    //double[] flattened = new double[m*n];
	    //System.out.println(flattened.length);
	    
	    //ix = 0;
	   // for (int i = 0; i < n; i++) {
	    	//for (int j = 0; j < m; j++) {
	    		//System.out.println(data[j][i]);
	    		//flattened[ix++] = data[j][i]; //col major flatten the array
	    	//}
	   // }
	    
	    
	    
	    
	    //Pointer RHSDensePtr = new Pointer(); //pointer for data of RHS dense matrix
	    //cudaMalloc(RHSDensePtr, m*n*Sizeof.DOUBLE); //mxn matrix
	    //int ldRHS = m; //leading dimension is m
	    //System.out.println("RHS CSR arrays");
	    //System.out.println(MatrixUtils.PointerContents(RHSRowCSRPtr, m+1, true));
	    //System.out.println(MatrixUtils.PointerContents(RHSColCSRPtr, nnzRHS[0], true));
	    //System.out.println(MatrixUtils.PointerContents(RHSValCSRPtr, nnzRHS[0], false));
	    //convert sparse representation to dense, needed for solver
	    //cusparseDcsr2dense(handle, m, n, descrRHS, RHSValCSRPtr, RHSRowCSRPtr, RHSColCSRPtr, RHSDensePtr, m);
	    //JCuda.cudaDeviceSynchronize(); //sync results up
	    
	    //cudaMemcpy(RHSDensePtr, Pointer.to(flattened), m*n*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    //cudaFree(RHSRowCSRPtr);
	    //cudaFree(RHSColCSRPtr);
	    //cudaFree(RHSValCSRPtr);
	    
	    /*
	     * Printing stuff for testing!!
	     */
	   // double printElement[] = new double[m*n];

	    //cudaMemcpy(Pointer.to(printElement), RHSDensePtr, m*n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix RHS(Dense): %d entries\n", printElement.length);
	    //ix = 0;
	    //for (int i = 0; i < n; i++) {
	    	//for (int j = 0; j < m; j++) {
	    		//System.out.printf("El: (%d,%d), Val: %f\n", j, i, printElement[ix++]);
	    	//}
	   //}
	    /*
	     * Printing stuff for testing!!
	     */
	    
	    /*
	     * Get Lu on the GPU, ready for the solver
	     */
	    
	    Pointer LuRowCooPtr = new Pointer(); //pointers for unseeded part of laplacian
	    Pointer LuRowCSRPtr = new Pointer();
	    Pointer LuColCSRPtr = new Pointer();
	    Pointer LuValCSRPtr = new Pointer();
	    
	    cudaMalloc(LuRowCooPtr, nnzLu*Sizeof.INT);//allocate memory for pointers
	    cudaMalloc(LuRowCSRPtr, (m+1)*Sizeof.INT);
	    cudaMalloc(LuColCSRPtr, nnzLu*Sizeof.INT);
	    cudaMalloc(LuValCSRPtr, nnzLu*Sizeof.DOUBLE);
	    
	    cudaMemcpy(LuRowCooPtr, Pointer.to(LuRowCooArr), nnzLu*Sizeof.INT, cudaMemcpyHostToDevice); //copy values in
	    cudaMemcpy(LuColCSRPtr, Pointer.to(LuColCooArr), nnzLu*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(LuValCSRPtr, Pointer.to(LuValCooArr), nnzLu*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    cusparseXcoo2csr(handle, LuRowCooPtr, nnzLu, pixelCount-num_seeds, LuRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO); //convert coo representation to csr representation
	    JCuda.cudaDeviceSynchronize(); //sync up
	    
	    /*
	     * Printing stuff for testing!!
	     *
	    printRow = new int[nnzLu];
	    printCol = new int[nnzLu];
	    printVal = new double[nnzLu];
	    
	    cudaMemcpy(Pointer.to(printRow), LuRowCooPtr, nnzLu*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printCol), LuColCSRPtr, nnzLu*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(printVal), LuValCSRPtr, nnzLu*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	    //System.out.printf("Matrix Lu: %d entries\n", nnzLu);
	    
	    for (int i = 0; i < nnzLu; i++) {
	    	//System.out.printf("Row: %d, Col: %d, Val: %f\n", printRow[i], printCol[i], printVal[i]);
	    }
	    /*
	     * Printing stuff for testing!!
	     */
	    
	    cudaFree(LuRowCooPtr);
	    
	    //System.out.println("Lu calculated and assigned on gpu");
	    
	    /*
	     * TESTING::::
	     *
	     */
	    /*double rhs[] = new double[m*n];
	    cudaMemcpy(Pointer.to(rhs), RHSDensePtr, m*n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    for (int i = 0; i < m*n; i++) {
	    	System.out.printf("RHS: %d, %f\n", i, rhs[i]);
	    }*/
	    
	    /*
	     * Set up and perform solving
	     * linear system is Lu*X=RHS
	     * where X is the matrix of probabilities
	     * use sparse solver
	     * http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrsmsolve
	     */
	    
	    cusparseMatDescr descrLu = new cusparseMatDescr(); //set up matrix description for Lu
	    cusparseCreateMatDescr(descrLu);
	    cusparseSetMatType(descrLu, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatIndexBase(descrLu, CUSPARSE_INDEX_BASE_ZERO);
	    
	    m = pixelCount-num_seeds; //height of solution
	    n = num_labels; //width of solution
	    
	    /*
	    Pointer ProbPtr = new Pointer();
	    Pointer alpha = new Pointer();
	    
	    double alphaArr[] = new double[1]; //array to hold alpha
	    alphaArr[0] = 1;
	    
	    //TODO: Current Point of Failure, solver does not work
	    //could we employ till's solver for each label? That takes square matrix, which Lu is not necessarily
	    
	    //compute LU decomposition, get L, U and P
	    //solve Ly = Pb
	    //solve Ux = y
	    //then output x???
	    
	    cudaMalloc(alpha, Sizeof.DOUBLE); //allocate memory for alpha
	    cudaMemcpy(alpha, Pointer.to(alphaArr), Sizeof.DOUBLE, cudaMemcpyHostToDevice); //copy value of alpha in (just 1 we dont want any multiplication)
	    
	    double Prob[] = new double[m*n]; //array to hold results
	    cudaMalloc(ProbPtr, (m*n)*Sizeof.DOUBLE); //allocate memory for results
	    cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo(); //create solver info
	    cusparseCreateSolveAnalysisInfo(info); //create on gpu
	    //System.out.println("Starting solve phase");
	    cusparseDcsrsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzLu, descrLu, LuValCSRPtr, LuRowCSRPtr, LuColCSRPtr, info); //perform analysis phase of solving
	    JCuda.cudaDeviceSynchronize();   
	    //System.out.println("Analysis complete");
	    cusparseDcsrsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, Pointer.to(alphaArr), descrLu, LuValCSRPtr, LuRowCSRPtr, LuColCSRPtr, info, RHSDensePtr, ldRHS, ProbPtr, m);
	    JCuda.cudaDeviceSynchronize();
	    //System.out.println("Solving complete");
	    cudaMemcpy(Pointer.to(Prob), ProbPtr, m*n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost); //copy results back
		*/
	    
	    //cudaFree(LuRowCSRPtr);
	    //cudaFree(LuColCSRPtr);
	    //cudaFree(LuValCSRPtr);
	    //cudaFree(ProbPtr);
	    //cudaFree(RHSDensePtr);
	    //cusparseDestroy(handle);
	    
	    
	    //System.out.println("System solved");
	    
	    /*
	     * Put output from solver into 2d array
	     * unseededpixels*labels, each entry has probability of pixel(row) being label(col)
	     */
	    //System.out.println("Output (X)");
	    
	    double out[][] = new double[num_labels][m];
	    /*/System.out.printf("%d\n", Prob.length);
	    for (int i = 0; i < num_labels; i++) { //print out array
	    	for (int j = 0; j < m; j++) {
	    		ix = i*m+j;
	    		//out[i][j] = Prob[ix];
	    		System.out.println(String.format("pixel no: %d, label no: %d, probability: %f", j, i, out[i][j]));
	    	}
	    }*/
	    
	    m = pixelCount - num_seeds; //m,n for rhs (and solution vector)
	    n = num_labels;	
	    
	    double[][] rhs = new double[n][m];
	    denseVector[] vects = new denseVector[n];
	    double[][] solution = new double[n][m];
	    
	    for (int i = 0; i < nnzRHS[0]; i++) {
	    	//System.out.println(RHSVal[i]);
	    	rhs[RHSCol[i]][RHSRow[i]] = RHSVal[i];
	    }
	    for (int i = 0; i < n; i++) {
	    	vects[i] = new denseVector(rhs[i]); //create a vector for each column of the right hand side of the equation
	    }
	    for (int i = 0; i < n; i++) { //solve for each column
	    	//System.out.println(MatrixUtils.PointerContents(vects[i].getPtr(), m, false));
	    	solution[i] = MatrixUtils.LuSolve(handle, descrLu, m, n, nnzLu, LuRowCSRPtr, LuColCSRPtr, LuValCSRPtr, vects[i], false);
	    	System.out.println("done");
	    }
	    
	    //for (int i = 0; i < n; i++) {
	    	//for (int j = 0; j < m; j++) {
	    		//System.out.print(solution[i][j]);
	    		//System.out.print(" ");
	    	//}
	    	//System.out.print("\n");
	    //}
	    //System.out.println("avg weight");
	    //System.out.println(avgweight);
	    
	    return solution;
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
	
	public static double weight(int a, int b, double beta) {
		
		/*
		 * Gets the weight of an edge (difference/similarity between two ends
		 */
		
		Color color1 = new Color(a);
		Color color2 = new Color(b);
		//System.out.printf("Getting weight between (%d, %d, %d, %d) and (%d, %d, %d, %d)\n",color1.getRed(), color1.getGreen(), color1.getBlue(), color1.getAlpha(), color2.getRed(), color2.getGreen(), color2.getBlue(), color2.getAlpha());
		//System.out.println("Getting weight between " + color1.toString() + " and " + color2.toString());
		int red = (int) Math.pow(color1.getRed() - color2.getRed(), 2);
		int green = (int) Math.pow(color1.getGreen() - color2.getGreen(), 2);
		int blue = (int) Math.pow(color1.getBlue() - color2.getBlue(), 2);
		//int alpha = (int) Math.pow(color1.getAlpha() - color2.getAlpha(), 2);
		int sum = red+green+blue;
		
		//int tot1 = Math.abs(color1.getRed()) + Math.abs(color1.getGreen()) + Math.abs(color1.getBlue());
		//int tot2 = Math.abs(color2.getRed()) + Math.abs(color2.getGreen()) + Math.abs(color2.getBlue());
		//tot1 /= 3;
		//tot2 /= 3;
		//System.out.print("weight ");
		//System.out.print(Math.exp(-beta*(Math.abs(tot1-tot2)/255)));
		//System.out.print("\n");
		//return (double) Math.exp(-beta*(Math.abs(tot1-tot2)/255));
		return (double) Math.exp(-(beta*sum));
		
		
	}
}