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
		int height = image.getHeight(); //get image height
		int pixelCount = width*height; //compute number of pixels
	    int edgeCount = (width*(height-1))+(height*(width-1)); //compute number of edges
	    
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
	    
	    //COO format
	    Pointer ARowCooPtr = new Pointer(); //pointer for row indices of A
	    Pointer AColCooPtr = new Pointer(); //pointer for column incices of A
	    Pointer AValCooPtr = new Pointer(); //pointer for values of A
	    
	    //CSR format
	    Pointer ARowCSRPtr = new Pointer(); //pointer for row indices of A
	    Pointer AColCSRPtr = new Pointer(); //pointer for column indices of A
	    Pointer AValCSRPtr = new Pointer(); //pointer for values of A
	    
	    //CSR format
	    Pointer A_tRowCSRPtr = new Pointer(); //pointer for row indices of A^T
	    Pointer A_tColCSRPtr = new Pointer(); //pointer for column indices of A^T
	    Pointer A_tValCSRPtr = new Pointer(); //pointer for values of A^T
	    
	    //COO format
	    Pointer CRowCooPtr = new Pointer(); //pointer for row indices of C
	    Pointer CColCooPtr = new Pointer(); //pointer for column indices of C
	    Pointer CValCooPtr = new Pointer(); //pointer for values of C
	    
	    //CSR format
	    Pointer CRowCSRPtr = new Pointer(); //pointer for row indices of C
	    Pointer CColCSRPtr = new Pointer(); //pointer for column indices of C
	    Pointer CValCSRPtr = new Pointer(); //pointer for values of C
	    
	    JCusparse.setExceptionsEnabled(true); //enable exceptions in jcuda
	    JCuda.setExceptionsEnabled(true);
	    
	    cusparseHandle handle = new cusparseHandle(); //create handle and matrix description
	    cusparseMatDescr descrA = new cusparseMatDescr();
	    cusparseMatDescr descrA_t = new cusparseMatDescr();
	    cusparseMatDescr descrC = new cusparseMatDescr();
	    
	    cudaMalloc(ARowCooPtr, nnzA*Sizeof.INT); //allocate pointer memory
	    cudaMalloc(AColCooPtr, nnzA*Sizeof.INT);
	    cudaMalloc(AValCooPtr, nnzA*Sizeof.DOUBLE);
	    
	    cudaMalloc(ARowCSRPtr, (nnzA+1)*Sizeof.INT);
	    cudaMalloc(AColCSRPtr, nnzA*Sizeof.INT);
	    cudaMalloc(AValCSRPtr, nnzA*Sizeof.DOUBLE);
	    
	    cudaMalloc(A_tRowCSRPtr, (nnzA+1)*Sizeof.INT);
	    cudaMalloc(A_tColCSRPtr, nnzA*Sizeof.INT);
	    cudaMalloc(A_tValCSRPtr, nnzA*Sizeof.DOUBLE);
	    
	    cudaMalloc(CRowCooPtr, nnzC*Sizeof.INT);
	    cudaMalloc(CColCooPtr, nnzC*Sizeof.INT);
	    cudaMalloc(CValCooPtr, nnzC*Sizeof.DOUBLE);
	    
	    cudaMalloc(CRowCSRPtr, (nnzC+1)*Sizeof.INT);
	    cudaMalloc(CColCSRPtr, nnzC*Sizeof.INT);
	    cudaMalloc(CValCSRPtr, nnzC*Sizeof.DOUBLE);
	    
	    //Create pointers pointing to arrays
	    
	    cudaMemcpy(ARowCooPtr, Pointer.to(ARowCoo), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(AColCooPtr, Pointer.to(AColCoo), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(AValCooPtr, Pointer.to(AValCoo), nnzA*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    cudaMemcpy(ARowCSRPtr, Pointer.to(ARowCSR), (nnzA+1)*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(AColCSRPtr, Pointer.to(AColCSR), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(AValCSRPtr, Pointer.to(AValCSR), nnzA*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    cudaMemcpy(A_tRowCSRPtr, Pointer.to(A_tRowCSR), (nnzA+1)*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(A_tColCSRPtr, Pointer.to(A_tColCSR), nnzA*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(A_tValCSRPtr, Pointer.to(A_tValCSR), nnzA*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    cudaMemcpy(CRowCooPtr, Pointer.to(CRowCoo), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(CColCooPtr, Pointer.to(CColCoo), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(CValCooPtr, Pointer.to(CValCoo), nnzC*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    cudaMemcpy(CRowCSRPtr, Pointer.to(CRowCSR), (nnzC+1)*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(CColCSRPtr, Pointer.to(CColCSR), nnzC*Sizeof.INT, cudaMemcpyHostToDevice);
	    cudaMemcpy(CValCSRPtr, Pointer.to(CValCSR), nnzC*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	    
	    
	    
	    
	    
	    int pixel[] = new int[pixelCount]; //array of pixels, numbered l-r t-b
	    Edge edge[] = new Edge[edgeCount]; //array of edges, numbered horizontally l-r followed by vertically t-b
	    
	    
	    for (int j = 0; j < width; j++) {
	      for (int i = 0; i < height; i++) {
	        pixel[j*height+i] = image.getRGB(i, j); //read in each pixel (l-r, t-b);
	      }
	    }
	    
	    
	    int edgeIndex = 0;//read in each edge, left-right
	    //horizontal edges
	    for (int j = 0; j < height; j++) {
	      for (int i = 0; i < width; i++) {
	        if (i != width-1) edge[edgeIndex++] = new Edge((j*height + i), (j*height + i + 1));
	      }
	    }
	    
	    //vertical edges
	    for (int i = 0; i < width; i++) { //read in each edge, top-bottom
	      for (int j = 0; j < height; j++) {
	        if (j != height-1) edge[edgeIndex++] = new Edge(j*height + i, (j+1)*height + i);
	      }
	    }
	    
	    //create matrix C in COO format
	    for (int i = 0; i < edgeCount; i++) {
	      CRowCoo[i] = edge[i].start;
	      CColCoo[i] = edge[i].end;
	      CValCoo[i] = weight(pixel[edge[i].start], pixel[edge[i].end]);
	    }
	    
	    int AEntries = 0; //create matrix A in COO format
	    for (int i = 0; i < edgeCount; i++) {
	      ARowCoo[AEntries] = edge[i].start;
	      AColCoo[AEntries] = i;
	      AValCoo[AEntries++] = -1;
	      ARowCoo[AEntries] = edge[i].end;
	      AColCoo[AEntries] = i;
	      AValCoo[AEntries++] = 1;
	    }
	    
	    cusparseCreate(handle);
	    
	    cusparseCreateMatDescr(descrA);
	    cusparseCreateMatDescr(descrA_t);
	    cusparseCreateMatDescr(descrC);
	    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrA_t, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
	    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrA_t, CUSPARSE_INDEX_BASE_ZERO);
	    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
	    
	    cusparseXcoo2csr(handle, ARowCooPtr, nnzA, nnzA/2, ARowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    AColCSR = AColCoo;
	    AValCSR = AValCoo;
	    cudaMemcpy(AColCSRPtr, Pointer.to(AColCSR), nnzA*Sizeof.INT, CUSPARSE_INDEX_BASE_ZERO);
	    cudaMemcpy(AValCSRPtr, Pointer.to(AValCSR), nnzA*Sizeof.DOUBLE, CUSPARSE_INDEX_BASE_ZERO);
	    
	    cusparseXcoo2csr(handle, AColCooPtr, nnzA, pixelCount, A_tRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    A_tColCSR = ARowCoo;
	    A_tValCSR = AValCoo;
	    cudaMemcpy(A_tColCSRPtr, Pointer.to(A_tColCSR), nnzA*Sizeof.INT, CUSPARSE_INDEX_BASE_ZERO);
	    cudaMemcpy(A_tValCSRPtr, Pointer.to(A_tValCSR), nnzA*Sizeof.DOUBLE, CUSPARSE_INDEX_BASE_ZERO);
	    
	    cusparseXcoo2csr(handle, CRowCooPtr, nnzC, nnzC, CRowCSRPtr, CUSPARSE_INDEX_BASE_ZERO);
	    CColCSR = CColCoo;
	    CValCSR = CValCoo;
	    cudaMemcpy(CColCSRPtr, Pointer.to(CColCSR), nnzC*Sizeof.INT, CUSPARSE_INDEX_BASE_ZERO);
	    cudaMemcpy(CValCSRPtr, Pointer.to(CValCSR), nnzC*Sizeof.DOUBLE, CUSPARSE_INDEX_BASE_ZERO);
	    
	    JCuda.cudaDeviceSynchronize();
	    
	    System.out.println();
	    for (int i = 0; i < nnzA; i++) {
	    	System.out.print(ARowCoo[i] + ", ");
	    }
	    System.out.println();
	    for (int i = 0; i < nnzA+1; i++) {
	    	System.out.print(ARowCSR[i] + ", ");
	    }
	    
	    
	    
	    cudaFree(ARowCooPtr);
	    cudaFree(AColCooPtr);
	    cudaFree(AValCooPtr);
	    
	    cudaFree(ARowCSRPtr);
	    cudaFree(AColCSRPtr);
	    cudaFree(AValCSRPtr);
	    
	    cudaFree(A_tRowCSRPtr);
	    cudaFree(A_tColCSRPtr);
	    cudaFree(A_tValCSRPtr);
	    
	    cudaFree(CRowCooPtr);
	    cudaFree(CColCooPtr);
	    cudaFree(CValCooPtr);
	   
	    cudaFree(CRowCSRPtr);
	    cudaFree(CColCSRPtr);
	    cudaFree(CValCSRPtr);
	    
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

