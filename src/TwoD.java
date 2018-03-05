import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.runtime.JCuda;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;


public class TwoD {
	
	public static double beta;

	public static void main(String[] args) {
		
		BufferedImage image = null;
		try { //read in image
			
			image = ImageIO.read(new File("bin/test_img.jpg"));
		
		} catch (Exception e) {
			
		}
		
		System.out.println("Enter beta: ");
		
		try { //read in beta value
			//beta = System.in.read();
			beta = 0.5;
		} catch (Exception E) {
			beta = 0.5;
		}
		
		int width = image.getWidth(); //get image width
		String out = String.format("width:%d", width);
		System.out.println(out);
		int height = image.getHeight(); //get image height
		out = String.format("height:%d", height);
		System.out.println(out);
		
		int pixelCount = width*height; //compute number of pixels
		out = String.format("Pixels:%d", pixelCount);
		System.out.println(out);
	    int edgeCount = (width*(height-1))+(height*(width-1)); //compute number of edges
	    out = String.format("Edges:%d", edgeCount);
	    System.out.println(out);
	    
	    int nnzC = edgeCount; //number of (non-zero) elements in matrix C (constitutive matrix)
	    int nnzA = edgeCount*2; //number of (non-zero) elements in matrix A (incidence matrix)
	    
	    
	    
	    int ARowCoo[] = new int[nnzA]; //arrays to hold matrix A in COO format
	    int AColCoo[] = new int[nnzA];
	    double AValCoo[] = new double[nnzA];
	    
	    int ARowCSR[] = new int[nnzA+1]; //arrays to hold matrix A in CSR format
	    int AColCSR[] = new int[nnzA];
	    double AValCSR[] = new double[nnzA];
	    
	    int A_tRowCSR[] = new int[nnzA+1]; //arrays to hold matrix A^T in CSR format
	    int A_tColCSR[] = new int[nnzA];
	    double A_tValCSR[] = new double[nnzA];
	    
	    int CRowCoo[] = new int[nnzC]; //arrays to hold matrix C in COO format
	    int CColCoo[] = new int[nnzC];
	    double CValCoo[] = new double[nnzC];
	    
	    int CRowCSR[] = new int[nnzC+1]; //arrays to hold matrix C in CSR format
	    int CColCSR[] = new int[nnzC];
	    double CValCSR[] = new double[nnzC];
	    	    
	    JCusparse.setExceptionsEnabled(true); //enable exceptions in jcuda
	    JCuda.setExceptionsEnabled(true);
	    
	    cusparseHandle handle = new cusparseHandle(); //create handle and matrix description
	    cusparseMatDescr descrA = new cusparseMatDescr();
	    cusparseMatDescr descrA_t = new cusparseMatDescr();
	    cusparseMatDescr descrC = new cusparseMatDescr();
	    cusparseMatDescr descrA_tC = new cusparseMatDescr();
	    cusparseMatDescr descrLap = new cusparseMatDescr();
	        
	    
	    int pixel[] = new int[pixelCount]; //array of pixels, numbered l-r t-b
	    Edge edge[] = new Edge[edgeCount]; //array of edges, numbered horizontally l-r followed by vertically t-b
	    
	    
	    for (int j = 0; j < width; j++) {
	      for (int i = 0; i < height; i++) {
	        pixel[j*height+i] = image.getRGB(i, j); //read in each pixel (l-r, t-b);
	        out = String.format("Pixel id %d", j*height+i);
	        //System.out.println(out);
	      }
	    }
	    
	    
	    int edgeIndex = 0;//read in each edge, left-right
	    //horizontal edges
	    for (int j = 0; j < height; j++) {
	      for (int i = 0; i < width; i++) {
	        if (i != width-1) {
	        	edge[edgeIndex++] = new Edge((j*height + i), (j*height + i + 1));
	        	out = String.format("Edge id: %d created", edgeIndex-1);
	        	System.out.println(out);
	        }
	      }
	    }
	    
	    //vertical edges
	    for (int i = 0; i < width; i++) { //read in each edge, top-bottom
	      for (int j = 0; j < height; j++) {
	        if (j != height-1) edge[edgeIndex++] = new Edge(j*height + i, (j+1)*height + i);
	        out = String.format("Edge id: %d created", edgeIndex-1);
        	System.out.println(out);
	      }
	    }
	    
	    //create matrix C in COO format
	    for (int i = 0; i < edgeCount; i++) {
	      CRowCoo[i] = i;
	      CColCoo[i] = i;
	      CValCoo[i] = weight(pixel[edge[i].start], pixel[edge[i].end]);
	      out = String.format("Constitutive entry at (%d, %d) with value %f created", CRowCoo[i], CColCoo[i], CValCoo[i]);
	      System.out.println(out);
	    }
	    
	    int AEntries = 0; //create matrix A in COO format
	    for (int i = 0; i < edgeCount; i++) {
	      ARowCoo[AEntries] = edge[i].start;
	      AColCoo[AEntries] = i;
	      AValCoo[AEntries++] = -1;
	      out = String.format("Incidence entry at (%d, %d) with value %f created", ARowCoo[AEntries-1], AColCoo[AEntries-1], AValCoo[AEntries-1]);
	      //System.out.println(out);
	      ARowCoo[AEntries] = edge[i].end;
	      AColCoo[AEntries] = i;
	      AValCoo[AEntries++] = 1;
	      out = String.format("Incidence entry at (%d, %d) with value %f created", ARowCoo[AEntries-1], AColCoo[AEntries-1], AValCoo[AEntries-1]);
	      //System.out.println(out);
	    }
	    
	    cusparseCreate(handle);
	    
	    cusparseCreateMatDescr(descrA);
	    cusparseCreateMatDescr(descrA_t);
	    cusparseCreateMatDescr(descrC);
	    cusparseCreateMatDescr(descrA_tC);
	    cusparseCreateMatDescr(descrLap);
	    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrA_t, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrA_t, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
	    
	    
	    //Get matrix A in CSR format on the GPU
	    Pointer ARowCooPtr = new Pointer(); //allocate pointers for GPU memory
	    Pointer ARowCSRPtr = new Pointer(); //hold rows for Coo and CSR representations
	    
	    cudaMalloc(ARowCooPtr, nnzA*Sizeof.INT); //allocate memory on GPU
	    cudaMalloc(ARowCSRPtr, (nnzA+1)*Sizeof.INT);
	    //copy COO representation into GPU memory
	    cudaMemcpy(ARowCooPtr, Pointer.to(ARowCoo), nnzA*Sizeof.INT, cudaMemcpyHostToDevice); 
	    //convert COO representation into CSR representation
	    cusparseXcoo2csr(handle, ARowCooPtr, nnzA, nnzA/2, ARowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    
	    AColCSR = AColCoo; //CSR of col is same as COO
	    AValCSR = AValCoo; //CSR of val is same as COO
	    //Garbage collection stuff
	    cudaFree(ARowCooPtr);
	    
	    
	    //Get matrix A^T in CSR format on the GPU
	    Pointer A_tRowCooPtr = new Pointer();
	    Pointer A_tRowCSRPtr = new Pointer();
	    //allocate memory on GPU
	    cudaMalloc(A_tRowCooPtr, nnzA*Sizeof.INT);
	    cudaMalloc(A_tRowCSRPtr, (nnzA+1)*Sizeof.INT);
	    //copy COO into GPU memory
	    cudaMemcpy(A_tRowCooPtr, Pointer.to(AColCoo), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    //convert to csr format
	    cusparseXcoo2csr(handle, A_tRowCooPtr, nnzA, nnzA/2, A_tRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    A_tColCSR = ARowCoo;
	    A_tValCSR = AValCoo;
	    //clean up stuff we dont need anymore
	    ARowCoo = null;
	    AColCoo = null;
	    AValCoo = null;
	    cudaFree(A_tRowCooPtr);
	    
	    
	    //Get matrix C in CSR format on the GPU
	    Pointer CRowCooPtr = new Pointer(); //allocate pointers for GPU memory
	    Pointer CRowCSRPtr = new Pointer();
	    
	    cudaMalloc(CRowCooPtr, nnzC*Sizeof.INT); //allocate memory on GPU
	    cudaMalloc(CRowCSRPtr, (nnzC+1)*Sizeof.INT);
	    //Copy COO representation into GPU memory
	    cudaMemcpy(CRowCooPtr, Pointer.to(CRowCoo), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
	    //convert COO representation into CSR representation
	    cusparseXcoo2csr(handle, CRowCooPtr, nnzC, nnzC/2, CRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    
	    CColCSR = CColCoo; //CSR Col is same as COO
	    CValCSR = CValCoo; //CSR val is same as COO
	    //garbage collection stuff
	    CColCoo = null; //null it to save memory
	    CValCoo = null; //null it to save memory
	    CRowCoo = null; //null it to save memory
	    cudaFree(CRowCooPtr); //free up the COO memory
	    
	    JCuda.cudaDeviceSynchronize();
	    //debug stuff rn
	    cudaMemcpy(Pointer.to(ARowCSR), ARowCSRPtr, (nnzA+1)*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(A_tRowCSR), A_tRowCSRPtr, (nnzA+1)*Sizeof.INT, cudaMemcpyDeviceToHost);
	    cudaMemcpy(Pointer.to(CRowCSR), CRowCSRPtr, (nnzC+1)*Sizeof.INT, cudaMemcpyDeviceToHost);
	   
	    for (int i = 0; i < nnzC; i++) {
	    	out = String.format("A:%d, A^T:%d, C:%d", ARowCSR[i], A_tRowCSR[i], CRowCSR[i]);
	    	System.out.println(out);
	    }
	    
	    Pointer AColCSRPtr = new Pointer(); //create column and value pointers on device memory
	    Pointer AValCSRPtr = new Pointer();
	    
	    cudaMalloc(AColCSRPtr, nnzA*Sizeof.INT); //allocate memory
	    cudaMalloc(AValCSRPtr, nnzA*Sizeof.DOUBLE);
	    
	    cudaMemcpy(AColCSRPtr, Pointer.to(AColCSR), nnzA*Sizeof.INT, cudaMemcpyHostToDevice); //copy values in to memory
	    cudaMemcpy(AValCSRPtr, Pointer.to(AValCSR), nnzA*Sizeof.DOUBLE, cudaMemcpyHostToDevice);  
	    
	    
	    Pointer A_tColCSRPtr = new Pointer();
	    Pointer A_tValCSRPtr = new Pointer();
	    
	    cudaMalloc(A_tColCSRPtr, nnzA*Sizeof.INT);
	    cudaMalloc(A_tValCSRPtr, nnzA*Sizeof.DOUBLE);
	    
	    cudaMemcpy(A_tColCSRPtr, Pointer.to(A_tColCSR), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(A_tValCSRPtr, Pointer.to(A_tValCSR), nnzA*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    
	    Pointer CColCSRPtr = new Pointer();
	    Pointer CValCSRPtr = new Pointer();
	    
	    cudaMalloc(CColCSRPtr, nnzC*Sizeof.INT);
	    cudaMalloc(CValCSRPtr, nnzC*Sizeof.DOUBLE);
	    
	    cudaMemcpy(CColCSRPtr, Pointer.to(CColCSR), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(CValCSRPtr, Pointer.to(CValCSR), nnzC*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    //perform first half of Laplacian calculation (A^t*C)
	    Pointer A_tCRowCSRPtr = new Pointer(); //allocate pointers for intermediate results
	    Pointer A_tCColCSRPtr = new Pointer();
	    Pointer A_tCValCSRPtr = new Pointer();
	    Pointer nnzA_tCPtr = new Pointer(); //pointer for nonzero elements
	    cudaMalloc(nnzA_tCPtr, Sizeof.INT); //allocate memory for nonzero elements
	    int nnzA_tC[] = new int[1]; //make it an array (just so we can copy back into it easier)
	    
	    int m,n,k; //A^T is mxk matrix, C is kxn matrix, A is kxm matrix
	    m = pixelCount; //values for m,n,k
	    k = edgeCount;
	    n = edgeCount;
	    //http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm 
	    //documentation for this section (shows how to use these methods)
	    cudaMalloc(A_tCRowCSRPtr, (m+1)*Sizeof.INT); //allocate memory for row pointer
	    //find row values and nnz for intermediate result
	    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA, ARowCSRPtr, AColCSRPtr, descrC, nnzC, CRowCSRPtr, CColCSRPtr, descrA_tC, A_tCRowCSRPtr, nnzA_tCPtr);
	    //copy nnz back to allocate memory for col and val pointers
	    cudaMemcpy(Pointer.to(nnzA_tC), nnzA_tCPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
	    //allocate memory
	    cudaMalloc(A_tCColCSRPtr, nnzA_tC[0]*Sizeof.INT);
	    cudaMalloc(A_tCValCSRPtr, nnzA_tC[0]*Sizeof.DOUBLE);
	    //perform second part of matrix multiplication
	    cusparseDcsrgemm(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA, nnzA, AValCSRPtr, ARowCSRPtr, AColCSRPtr, descrC, nnzC, CValCSRPtr, CRowCSRPtr, CColCSRPtr, descrA_tC, A_tCValCSRPtr, A_tCRowCSRPtr, A_tCColCSRPtr);
	    
	    
	    Pointer LapRowCSRPtr = new Pointer();
	    Pointer LapColCSRPtr = new Pointer();
	    Pointer LapValCSRPtr = new Pointer();
	    Pointer nnzLapPtr = new Pointer();
	    cudaMalloc(nnzLapPtr, Sizeof.INT);
	    int nnzLap[] = new int[1];
	    
	    //A^T*C is a mxk matrix, A is kxn matrix
	    m = pixelCount;
	    k = edgeCount;
	    n = pixelCount;
	    
	    cudaMalloc(LapRowCSRPtr, (m+1)*Sizeof.INT); 
	    
	    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA_tC, nnzA_tC[0], A_tCRowCSRPtr, A_tCColCSRPtr, descrA, nnzA, ARowCSRPtr, AColCSRPtr, descrLap, LapRowCSRPtr, nnzLapPtr);
	    
	    cudaMemcpy(Pointer.to(nnzLap), nnzLapPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
	    
	    cudaMalloc(LapColCSRPtr, nnzLap[0]*Sizeof.INT);
	    cudaMalloc(LapValCSRPtr, nnzLap[0]*Sizeof.DOUBLE);

	    cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrA_tC, nnzA_tC[0], A_tCValCSRPtr, A_tCRowCSRPtr, A_tCColCSRPtr, descrA, nnzA, AValCSRPtr, ARowCSRPtr, AColCSRPtr, descrLap, LapValCSRPtr, LapRowCSRPtr, LapColCSRPtr);
	    //form linear system and solve using cusparseDcsrsm_solve() 
	    //http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrsmsolve
	    
	    
	    cusparseDestroy(handle);
	    
		//can create laplacian matrix directly (requires degree for every node stored and searching by index for an edge)
		//very expensive
		//or we can have constitutive matrix, diagonal weights of each edge
		//and incidence matrix edge/node (need to assign edge orientation)
		//then L=A^T*C*A (maybe less expensive with gpu work)
		
		//potential at pixel vi for label s is x(s, i)
		
		//solve Lu*X = -B^T*M
		//Lu is laplacian for unseeded nodes
		//X has K columns, 1 for each x(s)
		//M has K columns for each m(s) m(s, j) = 1 if 
		
	}
	
	private static double weight(int a, int b) {
		
		Color color1 = new Color(a);
		Color color2 = new Color(b);
		
		int red = (int) Math.pow(color1.getRed() - color2.getRed(), 2);
		int green = (int) Math.pow(color1.getGreen() - color2.getGreen(), 2);
		int blue = (int) Math.pow(color1.getBlue() - color2.getBlue(), 2);
		int alpha = (int) Math.pow(color1.getAlpha() - color2.getAlpha(), 2);
		int sum = red+green+blue+alpha;
		
		return Math.exp(-(beta*sum));
		
		
	}

}

