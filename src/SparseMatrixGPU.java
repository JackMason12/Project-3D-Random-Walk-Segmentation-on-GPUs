import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;

public class SparseMatrixGPU { //sparse matrix on gpu
	
	public Pointer rowInd; //pointers
	public Pointer colInd;
	public Pointer vals;
	
	public cusparseMatDescr descr;
	
	public int nnz; //metadata about the matrix
	public int m;
	public int n;
	public String curFormat; //format of the matrix (1 of the 2 flags below)
	
	private String FLAG_COO = "coo"; //flags for matrix format
	private String FLAG_CSR = "csr";
	
	public SparseMatrixGPU(SparseMatrix mat) { //constructor from a normal sparse matrix
		
		this.nnz = mat.nnz; //set values
		this.m = mat.m;
		this.n= mat.n;		
		
		this.rowInd = new Pointer(); //declare the pointers
		this.colInd = new Pointer();
		this.vals = new Pointer();
		
		cudaMalloc(colInd, nnz*Sizeof.INT); //allocate memory and copy values in
		cudaMalloc(vals, nnz*Sizeof.DOUBLE);
		cudaMemcpy(this.colInd, Pointer.to(mat.colInd), this.nnz*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(this.vals, Pointer.to(mat.vals), this.nnz*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		
		if (mat.curFormat.equals(this.FLAG_COO)) { //set row indices pointer for COO format, and set format flag
			
			cudaMalloc(rowInd, nnz*Sizeof.INT);
			cudaMemcpy(this.rowInd, Pointer.to(mat.rowInd), this.nnz*Sizeof.INT, cudaMemcpyHostToDevice);
			this.curFormat = FLAG_COO;
			
		} else if (mat.curFormat.equals(this.FLAG_CSR)) { //set row indices pointer for CSR format, and set format flag
			
			cudaMalloc(rowInd, (m+1)*Sizeof.INT);
			cudaMemcpy(this.rowInd, Pointer.to(mat.rowInd), (this.m+1)*Sizeof.INT, cudaMemcpyHostToDevice);
			this.curFormat = FLAG_CSR;
			
		} else { //if the flag is invalid
			System.out.println("FLAG INCORRECT FOR SPARSE MATRIX GPU CREATION");
			return;
		}		
		InitDescr();
	}
	
	public SparseMatrixGPU(Pointer rowInd, Pointer colInd, Pointer vals, int nnz, int m, int n, String fmt, cusparseMatDescr desc) { //create from all the individual fields
		this.rowInd = rowInd; //set all the fields
		this.colInd = colInd;
		this.vals = vals;
		this.nnz = nnz;
		this.m = m;
		this.n = n;
		this.curFormat = fmt;
		this.descr = desc;
		System.out.println("Sparse mtx created on gpu with format " + fmt);
	}
	
	public void InitDescr() {
		this.descr = new cusparseMatDescr();
		cusparseCreateMatDescr(descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	}
	
	public SparseMatrixGPU ToCSR(cusparseHandle handle) { //return this matrix in csr format
		if (this.curFormat.equals(this.FLAG_CSR)) { //if already in the right format
			
			return this; //just return this
			
		} else { //use cusparse to convert to csr format
			
			Pointer csrRowInd = new Pointer(); //pointer for csr row
			cudaMalloc(csrRowInd, (this.m+1)*Sizeof.INT); //allocate mem for row
			cusparseXcoo2csr(handle, this.rowInd, this.nnz, this.m, csrRowInd, CUSPARSE_INDEX_BASE_ZERO); //convert to csr
			JCuda.cudaDeviceSynchronize(); //sync up
			return new SparseMatrixGPU(csrRowInd, this.colInd, this.vals, this.nnz, this.m, this.n, FLAG_CSR, this.descr); //return a new sparse matrix from the csr row
			
		}
	}
	
	public SparseMatrixGPU ToCOO(cusparseHandle handle) { //return this matrix in coo format
		if (this.curFormat.equals(this.FLAG_COO)) { //if already in right format
			
			return this; //just return this
			
		} else {  //use cusparse to convert to coo format
			
			Pointer cooRowInd = new Pointer(); //pointer for row
			cudaMalloc(cooRowInd, this.nnz*Sizeof.INT); //allocate memory
			cusparseXcsr2coo(handle, this.rowInd, this.nnz, this.m, cooRowInd, CUSPARSE_INDEX_BASE_ZERO); //convert to coo
			JCuda.cudaDeviceSynchronize(); //sync up
			return new SparseMatrixGPU(cooRowInd, this.colInd, this.vals, this.nnz, this.m, this.n, FLAG_COO, this.descr); //return new sparse mtx from coo row	
			
		}
	}
	
	public void ClearRow() { //clears out the row representation
		//safe to use after converting to a new format
		cudaFree(this.rowInd);
	}
	
	public void ClearAll() { //clears out everything
		//NOT SAFE to use after converting to a new format (col and val pointers are the same)
		cudaFree(this.rowInd);
		cudaFree(this.colInd);
		cudaFree(this.vals);
	}
	
	public SparseMatrix ToCPU() { //return this matrix on the cpu
		return new SparseMatrix(this);
	}
	
	public DenseMatrix ToDense() { //return this matrix as a dense matrix
		return this.ToCPU().ToDense();
	}
	
	public String toString() { //return string representation of this matrix
		return (new SparseMatrix(this)).toString();
		/*int rowSize = 0;
		if (this.curFormat.equals(this.FLAG_COO)) {
			rowSize = this.nnz;
		} else {
			rowSize = this.m + 1;
		}
		int[] rowInd = new int[rowSize];
		int[] colInd = new int[this.nnz];
		double[] vals = new double[this.nnz];
		
		cudaMemcpy(Pointer.to(rowInd), this.rowInd, rowSize*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(colInd), this.colInd, this.nnz*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(vals), this.vals, this.nnz*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		for (int i : rowInd) {
			System.out.print(i);
			System.out.print(",");
		}
		System.out.print("\n");
		for (int i : colInd) {
			System.out.print(i);
			System.out.print(",");
		}
		System.out.print("\n");
		for (double i : vals) {
			System.out.print(i);
			System.out.print(",");
		}
		
		return (new SparseMatrix(this.nnz, this.m, this.n, rowInd, colInd, vals, this.curFormat)).toString();
		*/
	}
}
