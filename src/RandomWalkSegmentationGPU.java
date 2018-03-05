import java.awt.Color;
import java.util.ArrayList;

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

public class RandomWalkSegmentationGPU {
	static double[][] getProbabilities(int[] pixels, int pixelCount, Edge[] edges, int edgeCount, double beta, int[] seeds, int[] labels) {
	
	    JCusparse.setExceptionsEnabled(true); //enable cusparse stuff
	    JCuda.setExceptionsEnabled(true);
	    
	    cusparseHandle handle = new cusparseHandle(); //create handle 
	    cusparseCreate(handle);
	    
		
		
		int nnzC = edgeCount; //number of nonzero elements in constitutive matrix
		//matrix C creation
	    int CRowCoo[] = new int[nnzC]; //arrays to hold matrix C in COO format
	    int CColCoo[] = new int[nnzC];
	    double CValCoo[] = new double[nnzC];
	    
	    for (int i = 0; i < nnzC; i++) { //populate COO representation of C
	    	CRowCoo[i] = i;//diagonal matrix of edge weights
	    	CColCoo[i] = i;
	    	CValCoo[i] = weight(pixels[edges[i].start], pixels[edges[i].end], beta);
	    }
	    
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
	    //free up coo representation of row
	    cudaFree(CRowCooPtr);
		
	    
	    
	    int nnzA = edgeCount*2; //number of nonzero elements in incidence matrix
	    //Matrix A creation
		int ARowCoo[] = new int[nnzA]; //arrays to hold matrix A in COO format
	    int AColCoo[] = new int[nnzA];
	    double AValCoo[] = new double[nnzA];
	    //populate matrix A
	    int AEntries = 0;
	    
	    for (int i = 0; i < edgeCount; i++) { //for each edge
	    	ARowCoo[AEntries] = edges[i].start;
	    	AColCoo[AEntries] = i;
	    	AValCoo[AEntries++] = -1; //assign edge an orientation -1 in source, +1 in dest
	    	ARowCoo[AEntries] = edges[i].end;
	    	AColCoo[AEntries] = i;
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
	    //free up row coo memory
		cudaFree(ARowCooPtr);
		
		
	    
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
	    
	    System.out.printf("nnzAtC: %d\n", nnzAtC[0]);
	    
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
	    
	    System.out.println("A^T*C calculated");
	    
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
	    
	    System.out.printf("nnzLap : %d\n", nnzLap[0]);
	    
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
	    
	    System.out.println("Laplacian calculated");
	    
	    
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
	    
	    /*System.out.println(nnzLap[0]);
	    for (int i = 0; i < nnzLap[0]; i++) {
	    	System.out.printf("%d,%d : %f\n", LapRowCoo[i], LapColCoo[i], LapValCoo[i]);
	    }*/
	    
	    System.out.println("Laplacian converted back to COO");
	    //free: LapRowCooPtr, LapRowCSRPtr, LapColCSRPtr, LapValCSRPtr
	    cudaFree(LapRowCooPtr);
	    cudaFree(LapRowCSRPtr);
	    cudaFree(LapColCSRPtr);
	    cudaFree(LapValCSRPtr);
	    
	    //coo representation for Lu (LHS of eq)
	    ArrayList<Integer> LuRowCoo = new ArrayList<Integer>();
	    ArrayList<Integer> LuColCoo = new ArrayList<Integer>();
	    ArrayList<Double> LuValCoo = new ArrayList<Double>();
	    
	    ArrayList<Integer> LbRowCoo = new ArrayList<Integer>(); //coo for Lb (part of RHS of eq)
	    ArrayList<Integer> LbColCoo = new ArrayList<Integer>();
	    ArrayList<Double> LbValCoo = new ArrayList<Double>();
	    System.out.println("ArrayLists created");
	    
	    for (int i = 0; i < nnzLap[0]; i++) { //populate array lists
	    	//System.out.println(i);
	    	LuRowCoo.add(LapRowCoo[i]); //pls give better solution
	    	LuColCoo.add(LapColCoo[i]); //I cant find a way to directly
	    	LuValCoo.add(LapValCoo[i]); //make an arraylist from an array
	    	LbRowCoo.add(LapRowCoo[i]);
	    	LbColCoo.add(LapColCoo[i]);
	    	LbValCoo.add(LapValCoo[i]);	    	
	    }
	    System.out.println("ArrayLists populated");
	     
	    int ix; //index
	    int nnzLu = nnzLap[0]; //num nonzero for Lu and Lb
	    int nnzLb = nnzLap[0];
	    
	    for (int i = 0; i < num_seeds; i++) { //remove seed rows and cols to get unseeded part of laplacian for problem
	    	ix = LuRowCoo.indexOf(seeds[i]); //Lu
	    	while (ix != -1) {
	    		LuRowCoo.remove(ix);
	    		LuColCoo.remove(ix);
	    		LuValCoo.remove(ix);	  
	    		nnzLu--;
	    		ix = LuRowCoo.indexOf(seeds[i]);
	    	}
	    	ix = LuColCoo.indexOf(seeds[i]);
	    	while (ix != -1) {
	    		LuRowCoo.remove(ix);
	    		LuColCoo.remove(ix);
	    		LuValCoo.remove(ix);
	    		nnzLu--;
	    		ix = LuColCoo.indexOf(seeds[i]);
	    	}
	    }
	    
	    Integer LuRowCooArrTemp[] = LuRowCoo.toArray(new Integer[nnzLu]); //get back into arrays
	    Integer LuColCooArrTemp[] = LuColCoo.toArray(new Integer[nnzLu]);
	    Double LuValCooArrTemp[] = LuValCoo.toArray(new Double[nnzLu]);
	    
	    int LuRowCooArr[] = new int[nnzLu]; //put back in to proper primitive types
	    int LuColCooArr[] = new int[nnzLu];
	    double LuValCooArr[] = new double[nnzLu];
	    
	    for (int i = 0; i < nnzLu; i++) {
	    	LuRowCooArr[i] = LuRowCooArrTemp[i];
	    	LuColCooArr[i] = LuColCooArrTemp[i];
	    	LuValCooArr[i] = LuValCooArrTemp[i].doubleValue();
	    }
	    
	    
	    
	    ArrayList<Integer> seed = new ArrayList<Integer>(); //get seeds in arraylist
	    for (int i = 0; i < num_seeds; i++) { //get seeds in arraylist
	    	seed.add(seeds[i]); //just so we get .contains method
	    }
	    for (int i = 0; i < pixelCount; i++) { //Lb (boundary values
	    	if (seed.contains(i)) { //remove seed rows, and unseeded columns
	    		ix = LbRowCoo.indexOf(i);
	    		while(ix != -1) {
	    			LbRowCoo.remove(ix);
	    			LbColCoo.remove(ix);
	    			LbValCoo.remove(ix);
	    			nnzLb--;
	    			ix = LbRowCoo.indexOf(i);
	    		}
	    	} else {
	    		ix = LbColCoo.indexOf(i);
	    		while (ix != -1) {
	    			LbRowCoo.remove(ix);
	    			LbColCoo.remove(ix);
	    			LbValCoo.remove(ix);
	    			nnzLb--;
	    			ix = LbColCoo.indexOf(i);
	    		}
	    	}
	    }
	    
	    m = num_seeds; //num rows of b
	    n = num_labels; //num cols of b
	    
	    int bRowCoo[] = new int[m]; //arrays to hold b in coo format
	    int bColCoo[] = new int[m];
	    double bValCoo[] = new double[m];
	    
	    for (int i = 0; i < num_seeds; i++) { //set up b
	    	bRowCoo[i] = i;
	    	bColCoo[i] = labels[i];
	    	bValCoo[i] = 1;
	    }
	    
	    Integer LbRowCooArrTemp[] = LbRowCoo.toArray(new Integer[nnzLb]); //set up boundary value laplacian
	    Integer LbColCooArrTemp[] = LbColCoo.toArray(new Integer[nnzLb]);
	    Double LbValCooArrTemp[] = LbValCoo.toArray(new Double[nnzLb]);
	    int LbRowCooArr[] = new int[nnzLb];
	    int LbColCooArr[] = new int[nnzLb];
	    double LbValCooArr[] = new double[nnzLb];
	    for (int i = 0; i < nnzLb; i++) {
	    	LbRowCooArr[i] = LbRowCooArrTemp[i]; 
	    	LbColCooArr[i] = LbColCooArrTemp[i];
	    	LbValCooArr[i] = -LbValCooArrTemp[i].doubleValue();//value is neg because we want -Lb*b
	    }
	    
	    //allocate matrix b on GPU
	    Pointer bRowCooPtr = new Pointer();//pointers
	    Pointer bRowCSRPtr = new Pointer();
	    Pointer bColCSRPtr = new Pointer();
	    Pointer bValCSRPtr = new Pointer();
	    
	    int nnzB = num_seeds; //num non zero elements
	    
	    cudaMalloc(bRowCooPtr, nnzB*Sizeof.INT); //allocate memory for pointers
	    cudaMalloc(bRowCSRPtr, (pixelCount-num_seeds+1)*Sizeof.INT);
	    cudaMalloc(bColCSRPtr, nnzB*Sizeof.INT);
	    cudaMalloc(bValCSRPtr, nnzB*Sizeof.DOUBLE);
	    
	    cudaMemcpy(bRowCooPtr, Pointer.to(bRowCoo), nnzB*Sizeof.INT, cudaMemcpyHostToDevice);//copy values in to pointers
	    cudaMemcpy(bColCSRPtr, Pointer.to(bColCoo), nnzB*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(bValCSRPtr, Pointer.to(bValCoo), nnzB*Sizeof.DOUBLE, cudaMemcpyHostToDevice);	    
	    
	    cusparseXcoo2csr(handle, bRowCooPtr, nnzB, nnzB, bRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO); //convert COO row representation to CSR (for multiplication later)
	    JCuda.cudaDeviceSynchronize(); //sync up
	    cudaFree(bRowCooPtr);//free up unneeded memory

	    
	    System.out.println("b calculated and assigned on gpu");
	    
	    m = num_seeds;
	    k = pixelCount-num_seeds;
	    
	    Pointer LbRowCooPtr = new Pointer(); //form matrix lb
	    Pointer LbRowCSRPtr = new Pointer(); //assign on gpu
	    Pointer LbColCSRPtr = new Pointer(); //create pointers
	    Pointer LbValCSRPtr = new Pointer();
	    
	    cudaMalloc(LbRowCooPtr, nnzLb*Sizeof.INT); //allocate memory for pointers
	    cudaMalloc(LbRowCSRPtr, (k+1)*Sizeof.INT);
	    cudaMalloc(LbColCSRPtr, nnzLb*Sizeof.INT);
	    cudaMalloc(LbValCSRPtr, nnzLb*Sizeof.DOUBLE);
	    
	    cudaMemcpy(LbRowCooPtr, Pointer.to(LbRowCooArr), nnzLb*Sizeof.INT, cudaMemcpyHostToDevice); //copy values into pointer
	    cudaMemcpy(LbColCSRPtr, Pointer.to(LbColCooArr), nnzLb*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(LbValCSRPtr, Pointer.to(LbValCooArr), nnzLb*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    cusparseXcoo2csr(handle, LbRowCooPtr, nnzLb, m, LbRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO); //convert COO representation into csr representation
	    JCuda.cudaDeviceSynchronize(); //synchronise
	    //cudaFree(LbRowCooPtr);
	    
	    //TODO: seems like every laplacian element is 0
	    /*cudaMemcpy(Pointer.to(LbRowCoo), LbRowCooPtr, nnzLb*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(LbColCoo), LbColCSRPtr, nnzLb*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(LbValCoo), LbValCSRPtr, nnzLb*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    for (int i = 0; i < nnzLb; i++) {
	    	System.out.printf("%d, %d : %f\n", LbRowCoo[i], LbColCoo[i], LbValCoo[i]);
	    }*/
	    
	    
	    System.out.println("Lb calculated and assigned on gpu");
	    
	    Pointer RHSRowCSRPtr = new Pointer(); //right hand side of problem
	    Pointer RHSColCSRPtr = new Pointer(); // is -lb*b
	    Pointer RHSValCSRPtr = new Pointer(); //pointers for right hand side as sparse matrix (use sparse matrix multipication)
	    Pointer nnzRHSPtr = new Pointer();
	    
	    int nnzRHS[] = new int[1];
	    m = pixelCount - num_seeds; //m,n,k for multiplication -Lb*b
	    k = num_seeds; //Lb is m*k matrix
	    n = num_labels; //b is k*n matrix
	    
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
	    
	    cudaMalloc(RHSRowCSRPtr, (m+1)*Sizeof.INT); //create RHS
	    cudaMalloc(nnzRHSPtr, Sizeof.INT); //allocate memory for rhs row
	    
	    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrLb, nnzLb, LbRowCSRPtr, LbColCSRPtr, descrb, nnzB, bRowCSRPtr, bColCSRPtr, descrRHS, RHSRowCSRPtr, nnzRHSPtr);
	    JCuda.cudaDeviceSynchronize(); //find nnz for rhs, synchronise
	    cudaMemcpy(Pointer.to(nnzRHS), nnzRHSPtr, Sizeof.INT, cudaMemcpyDeviceToHost); //copy nnz back to host
	    cudaFree(nnzRHSPtr);
	    
	    cudaMalloc(RHSColCSRPtr, nnzRHS[0]*Sizeof.INT); //allocate memory for col and value pointers
	    cudaMalloc(RHSValCSRPtr, nnzRHS[0]*Sizeof.DOUBLE);
	    
	    cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrLb, nnzLb, LbValCSRPtr, LbRowCSRPtr, LbColCSRPtr, descrb, nnzB, bValCSRPtr, bRowCSRPtr, bColCSRPtr, descrRHS, RHSValCSRPtr, RHSRowCSRPtr, RHSColCSRPtr);
	    JCuda.cudaDeviceSynchronize(); //perform multiplication and synchronise
	    System.out.printf("nnzRHS : %d\n", nnzRHS[0]);
	    
	    
	    /*int RHSCol[] = new int[nnzRHS[0]];
	    int RHSRow[] = new int[nnzRHS[0]];
	    double RHSVal[] = new double[nnzRHS[0]];
	    cudaMemcpy(Pointer.to(RHSCol), RHSColCSRPtr, m*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(RHSRow), RHSRowCSRPtr, (m+1)*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(RHSVal), RHSValCSRPtr, m*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    for (int i = 0; i < nnzRHS[0]; i++) {
	    	System.out.println(String.format("%d,%d : %f", RHSRow[i], RHSCol[i], RHSVal[i]));
	    }*/
	    
	    cudaFree(bRowCSRPtr); //free up pointers that are no-longer needed
	    cudaFree(bColCSRPtr);
	    cudaFree(bValCSRPtr);
	    cudaFree(LbRowCSRPtr);
	    cudaFree(LbColCSRPtr);
	    cudaFree(LbValCSRPtr);
	    
	    System.out.println("RHS calculated and assigned on gpu");
	    
	    //eq will be Lu*X=-Lb*Bound where bound is a seedcount*labelcount matrix, where an entry is 1 if that seed (row) has the same label as the col
	    
	    Pointer RHSDensePtr = new Pointer(); //pointer for data of RHS dense matrix
	    cudaMalloc(RHSDensePtr, m*n*Sizeof.DOUBLE); //mxn matrix
	    int ldRHS = m; //leading dimension is m
	    
	    //convert sparse representation to dense, needed for solver
	    cusparseDcsr2dense(handle, m, n, descrRHS, RHSValCSRPtr, RHSRowCSRPtr, RHSColCSRPtr, RHSDensePtr, ldRHS);
	    JCuda.cudaDeviceSynchronize(); //sync results up
	    cudaFree(RHSRowCSRPtr);
	    cudaFree(RHSColCSRPtr);
	    cudaFree(RHSValCSRPtr);
	    
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
	    
	    cudaFree(LuRowCooPtr);
	    
	    System.out.println("Lu calculated and assigned on gpu");
	    
	    cusparseMatDescr descrLu = new cusparseMatDescr(); //set up matrix description for Lu
	    cusparseCreateMatDescr(descrLu);
	    cusparseSetMatType(descrLu, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatIndexBase(descrLu, CUSPARSE_INDEX_BASE_ZERO);
	    
	    m = pixelCount-num_seeds; //height of solution
	    n = num_labels; //width of solution
	    
	    Pointer ProbPtr = new Pointer();
	    Pointer alpha = new Pointer();
	    
	    double alphaArr[] = new double[1]; //array to hold alpha
	    alphaArr[0] = 1;
	    
	    cudaMalloc(alpha, Sizeof.DOUBLE); //allocate memory for alpha
	    cudaMemcpy(alpha, Pointer.to(alphaArr), Sizeof.DOUBLE, cudaMemcpyHostToDevice); //copy value of alpha in (just 1 we dont want any multiplication)
	    
	    double Prob[] = new double[m*n]; //array to hold results
	    cudaMalloc(ProbPtr, (m*n)*Sizeof.DOUBLE); //allocate memory for results
	    cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo(); //create solver info
	    cusparseCreateSolveAnalysisInfo(info); //create on gpu
	    System.out.println("Starting solve phase");
	    cusparseDcsrsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzLu, descrLu, LuValCSRPtr, LuRowCSRPtr, LuColCSRPtr, info); //perform analysis phase of solving
	    JCuda.cudaDeviceSynchronize();
	    
	    /*double RHS[] = new double[ldRHS*n];
	    cudaMemcpy(Pointer.to(RHS), RHSDensePtr, ldRHS*n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    System.out.println(ldRHS);
	    for (int i = 0; i < ldRHS*n; i++) {
	    	System.out.println(RHS[i]);
	    }
	    */    
	    System.out.println("Analysis complete");
	    //perform solving (crashes but I dont know why)
	    
	    System.out.println(alpha);
	    System.out.println(info);
	    System.out.println(LuValCSRPtr);
	    System.out.println(LuRowCSRPtr);
	    System.out.println(LuColCSRPtr);
	    System.out.println(RHSDensePtr);
	    System.out.println(ProbPtr);
	    
	    
	    
	    cusparseDcsrsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, Pointer.to(alphaArr), descrLu, LuValCSRPtr, LuRowCSRPtr, LuColCSRPtr, info, RHSDensePtr, ldRHS, ProbPtr, m);
	    System.out.println("Solving complete");
	    cudaMemcpy(Pointer.to(Prob), ProbPtr, m*n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost); //copy results back
		
	    
	    cudaFree(LuRowCSRPtr);
	    cudaFree(LuColCSRPtr);
	    cudaFree(LuValCSRPtr);
	    cudaFree(ProbPtr);
	    cudaFree(RHSDensePtr);
	    cusparseDestroy(handle);
	    
	    
	    System.out.println("System solved");
	    
	    double out[][] = new double[num_labels][m];
	    System.out.printf("%d\n", Prob.length);
	    for (int i = 0; i < num_labels; i++) { //print out array
	    	for (int j = 0; j < m; j++) {
	    		ix = i*m+j;
	    		out[i][j] = Prob[ix];
	    		System.out.println(Prob[ix]);
	    		System.out.println(String.format("label no: %d, pixel no: %d, probability: %f", i, j, out[i][j]));
	    	}
	    }
	    
	    
	    
	    return out;
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
}
