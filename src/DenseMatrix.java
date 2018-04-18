public class DenseMatrix {
	public int m;
	public int n;
	public int ld;
	public double[][] data;
	
	public DenseMatrix(int m, int n, double[][] data) {
		this.m = m;
		this.n = n;
		this.ld = m;
		this.data = data;
	}
	
	public DenseMatrix(SparseMatrix mat) {
		DenseMatrix tmp = mat.ToDense();
		this.m = tmp.m;
		this.n= tmp.n;
		this.ld = tmp.ld;
		this.data = tmp.data;		
	}
	
	public DenseMatrix(SparseMatrixGPU mat) {
		DenseMatrix tmp = mat.ToCPU().ToDense();
		this.m = tmp.m;
		this.n = tmp.n;
		this.ld = tmp.ld;
		this.data = tmp.data;
	}
	
	public DenseMatrix Transpose() { //returns transpose of this matrix
		double[][] out = new double[this.n][this.m];
		for (int i = 0; i < this.m; i++) {
			for (int j = 0; j < this.n; j++) {
				out[j][i] = this.data[i][j]; 
			}
		}
		return new DenseMatrix(this.n, this.m, out);
	}
	
	public SparseMatrix ToSparse() {
		int nnz = 0;
		for (int i = 0; i < this.m; i++) {
			for (int j = 0; j < this.n; j++) {
				if (this.data[i][j] != 0) nnz++;
			}
		}
		int[] rowInd = new int[nnz];
		int[] colInd = new int[nnz];
		double[] vals = new double[nnz];
		int index = 0;
		for (int j = 0; j < this.n; j++) {
			for (int i = 0; i < this.m; i++) {
				if (this.data[i][j] != 0) {
					rowInd[index] = j;
					colInd[index] = i;
					vals[index++] = this.data[i][j];					
				}
			}
		}
		return new SparseMatrix(nnz, this.m, this.n, rowInd, colInd, vals, "coo");
	}
	
	public String toString() {
		String out = "";
		for (int i = 0; i < this.m; i++) { //print out this representation
			for (int j = 0; j < this.n; j++) {
				out += this.data[i][j];
				out += " ";
			}
			out += '\n';
		}
		return out;
	}
	
}
