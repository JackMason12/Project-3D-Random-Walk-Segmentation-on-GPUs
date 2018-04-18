import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import static jcuda.runtime.cudaMemcpyKind.*;

import java.awt.Color;
import java.awt.image.BufferedImage;

import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;

public class GPURW {
	
	private int[] pixels;
	private Edge[] edges;
	private double[] weights;
	private boolean initialised = false;
	private double beta;
	
	public DenseMatrix GetProbabilities(int[] seeds, int[] labels, int labelCount) {
		if (!this.initialised) {
			System.out.println("Instance not initialised");
		}
		if (edges.length != weights.length) {
			System.out.println("Not the same number of edges and weights");
		}
		int pixelCount = pixels.length;
		int edgeCount = edges.length;
		int seedCount = seeds.length;
		
		SparseMatrix cCOO = new SparseMatrix(edgeCount, edgeCount, edgeCount, "coo");
		cCOO.PopulateDiagonal(this.weights);
		System.out.println("C - COO");
		System.out.println(cCOO.toString());
		SparseMatrix cCSR = cCOO.ToCSR();
		System.out.println("C - CSR");
		System.out.println(cCSR.toString());
		//cCOO.Clear();
		
		SparseMatrix aCOO = new SparseMatrix(edgeCount*2, edgeCount, pixelCount, "coo");
		aCOO.PopulateIncidence(edges);
		System.out.println("A - COO");
		System.out.println(aCOO.toString());
		SparseMatrix atCSR = aCOO.Transpose().ToCSR();
		System.out.println("A^T - CSR");
		System.out.println(atCSR.toString());
		SparseMatrix aCSR = aCOO.ToCSR();
		System.out.println("A - CSR");
		System.out.println(aCSR.toString());
		//aCOO.Clear();
		
		JCusparse.setExceptionsEnabled(true);
		JCuda.setExceptionsEnabled(true);
		
		cusparseHandle handle = new cusparseHandle();
		cusparseCreate(handle);
		
		SparseMatrixGPU c = new SparseMatrixGPU(cCSR);
		System.out.println("C - GPU");
		System.out.println(c.toString());
		SparseMatrixGPU a = new SparseMatrixGPU(aCSR);
		System.out.println("A - GPU");
		System.out.println(a.toString());
		SparseMatrixGPU at = new SparseMatrixGPU(atCSR);
		System.out.println("A^T - GPU");
		System.out.println(at.toString());
		
		SparseMatrixGPU atc = MatrixUtils.Multiply(at, c, handle);
		System.out.println("A^T * C - GPU");
		System.out.println(atc.toString());
		SparseMatrixGPU Lap = MatrixUtils.Multiply(atc, a, handle);
		System.out.println("Lap - GPU");
		System.out.println(Lap.toString());
		
		return null;
	}
	
	public void Init1D(BufferedImage img) {
		
	}
	
	public void Init2D(BufferedImage img) {
		
	}
	
	public void Init3D(BufferedImage img) {
		
	}
	
	public void InitFromArrays(int[] pixelsIn, Edge[] edgesIn) { //initialise from arrays 
		//no beta given
		this.pixels = new int[pixelsIn.length];
		this.edges = new Edge[edgesIn.length];
		this.weights = new double[edgesIn.length];
		this.pixels = pixelsIn;
		this.edges = edgesIn;
		for (int i = 0; i < edgesIn.length;i++) {
			this.weights[i] = GetWeight(this.edges[i]);
		}
		this.beta = 0.5;
		this.initialised = true;
	}
	
	public void InitFromArrays(int[] pixelsIn, Edge[] edgesIn, double beta) { //initialise from arrays
		//beta given
		this.pixels = new int[pixelsIn.length];
		this.edges = new Edge[edgesIn.length];
		this.weights = new double[edgesIn.length];
		this.pixels = pixelsIn;
		this.edges = edgesIn;
		for (int i = 0; i < edgesIn.length;i++) {
			this.weights[i] = GetWeight(this.edges[i]);
		}
		this.beta = beta;
		this.initialised = true;
	}
	
	private double GetWeight(Edge edge) { //get the weight of an edge (difference between 2 pixels)
		Color c1 = new Color(this.pixels[edge.start]); //create colours for each pixel
		Color c2 = new Color(this.pixels[edge.end]);
		
		double red = Math.sqrt(Math.pow(c1.getRed()-c2.getRed(), 2)); //get difference between colour components
		double green = Math.sqrt(Math.pow(c1.getGreen()-c2.getGreen(), 2));
		double blue = Math.sqrt(Math.pow(c1.getBlue()-c2.getBlue(), 2));
		
		double diff = red+green+blue/3; //average difference between colour components
		double multi = this.beta/255; //multiplier (divide by 255 to get normalised)
		double w = Math.exp(-multi*diff); //e^-b*diff
		
		return w; //return weight
	}
}
