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

public class SparseMatrix { //structure for sparse matrices
	
	public int nnz; //matrix metadata (mxn matrix, nnz elements)
	public int m;
	public int n;
	public String curFormat; //format of matrix (coo or csr)
	
	public int[] colInd; //arrays to hold matrix
	public int[] rowInd;
	public double[] vals;	
	
	private String FLAG_COO = "coo";
	private String FLAG_CSR = "csr";
	
	public SparseMatrix(int nnz, int m, int n, String flag) { //create empty sparse matrix 
		this.nnz = nnz;
		this.m = m;
		this.n = n;
		this.colInd = new int[nnz];
		this.vals = new double[nnz];
		if (flag.equals(this.FLAG_COO)) {
			this.rowInd = new int[nnz];
			this.curFormat = FLAG_COO;
		} else if (flag.equals(this.FLAG_CSR)) {
			this.rowInd = new int[m+1];
			this.curFormat = FLAG_CSR;
		} else {
			System.out.println("INVALID FLAG IN SPARSE MATRIX INITIALISATION");
		}
	}
	
	public SparseMatrix(int nnz, int m, int n, int[] rowInd, int[] colInd, double[] vals, String fmt) {
		this.nnz = nnz;
		this.m = m;
		this.n = n;
		this.rowInd = rowInd;
		this.colInd = colInd;
		this.vals = vals;
		this.curFormat = fmt;
	}
	
	public SparseMatrix(SparseMatrixGPU mat) {
		int rowSize = 0;
		if (mat.curFormat.equals(this.FLAG_COO)) {
			rowSize = mat.nnz;
		} else {
			rowSize = mat.m + 1;
		}
		this.nnz = mat.nnz;
		this.m = mat.m;
		this.n = mat.n;
		this.curFormat = mat.curFormat;
		this.rowInd = new int[rowSize];
		this.colInd = new int[this.nnz];
		this.vals = new double[this.nnz];
		
		cudaMemcpy(Pointer.to(this.rowInd), mat.rowInd, rowSize*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(this.colInd), mat.colInd, this.nnz*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(this.vals), mat.vals, this.nnz*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);	
	}
	
	public void PopulateDiagonal(double[] values) { //populate as a diagonal matrix
		for (int i = 0; i < values.length; i++) { //populate as expected
			this.colInd[i] = i; //actually works for csr and coo
			this.rowInd[i] = i;
			this.vals[i] = values[i];
		}
	}
	
	public void PopulateIncidence(Edge[] edges) { //populates sparse matrix as incidence matrix (only works for coo format)
		int entries = 0;
		for (int i = 0; i < edges.length; i++) { //populate as incidence matrix from array of edges
			this.colInd[entries] = edges[i].start;
			this.rowInd[entries] = i;
			this.vals[entries++] = -1;
			this.colInd[entries] = edges[i].end;
			this.rowInd[entries] = i;
			this.vals[entries++] = 1;
		}
	}
	
	public SparseMatrix Transpose() { //return the transpose of this matrix
		if (this.curFormat.equals(this.FLAG_CSR)) {
			return this.ToCOO().Transpose().ToCSR();
		} else {
			return this.ToDense().Transpose().ToSparse();
		}
	}
	
	public SparseMatrix ToCSR() { //return this matrix as a csr matrix
		if (this.curFormat.equals(this.FLAG_COO)) {
			
			int[] csrRowInd = new int[this.m + 1];
			int[] nnzpr = new int[this.m];
			for (int i = 0; i < this.nnz; i++) { //get nnz for each row
				nnzpr[this.rowInd[i]]++;
			}
			csrRowInd[0] = 0;
			int tot = 0;
			for (int i = 0; i < this.m; i++) { //form csr row arrays
				tot += nnzpr[i];
				csrRowInd[i+1] = tot;
			}
			return new SparseMatrix(this.nnz, this.m, this.n, csrRowInd, this.colInd, this.vals, FLAG_CSR);
			
		} else {
			return this;
		}
	}
	
	public SparseMatrix ToCOO() { //return this matrix as a coo matrix
		if (this.curFormat.equals(this.FLAG_COO)) {
			return this;
		} else {
			
			int[] cooRowInd = new int[this.nnz];
			int cur = 0;
			int ix = 0;
			for (int i = 1; i < this.m + 1; i++) { //for each row i
				cur = this.rowInd[i] - this.rowInd[i-1]; //get the number of elements in row i-1
				for (int j = 0; j < cur; j++) { //set this amount of elements in this row
					cooRowInd[ix + j] = i-1;
				}
				ix = ix + cur; //increment index
			}
			
			return new SparseMatrix(this.nnz, this.m, this.n, this.colInd, cooRowInd, this.vals, this.FLAG_COO);
			
		}
	}
	
	public SparseMatrixGPU ToGPU() { //returns this matrix as a matrix on the gpu
		return new SparseMatrixGPU(this);
	}
	
	public DenseMatrix ToDense() {
		double[][] data = new double[this.m][this.n];
		if (this.curFormat.equals(FLAG_CSR)) {
			return this.ToCOO().ToDense();
		} else {
			System.out.println(data[0].length);
			System.out.println(data.length);
			for (int i = 0; i < this.nnz; i++) {
				System.out.printf("(%d,%d)\n", this.colInd[i], this.rowInd[i]);
				data[this.rowInd[i]][this.colInd[i]] = this.vals[i];
			}
		}
		return new DenseMatrix(this.m, this.n, data);
	}
	
	public void Clear() { //0 out everything (hopefully cleans up memory)
		this.nnz = 0;
		this.m = 0;
		this.n = 0;
		this.curFormat = null;		
		this.colInd = new int[0]; 
		this.rowInd = new int[0];
		this.vals = new double[0];	
	}
	
	public String toString() { //return string representation of this matrix
		return this.ToDense().toString();
	}
	
}
