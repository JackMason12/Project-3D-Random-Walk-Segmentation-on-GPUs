import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.*;


public class denseVector {
	
	/*
	 * Written by Till Bretschneider
	 * Simple storage class for vectors on the GPU
	 * 
	 */

	private Pointer v_gpuPtr = new Pointer();
	private int size;

	public denseVector(double[] v){
		// creates dense Vector from double array
		size = v.length;

		cudaMalloc(v_gpuPtr, size*Sizeof.DOUBLE);
		cudaMemcpy(v_gpuPtr, Pointer.to(v), size*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	}

	public denseVector(int n){
		// allocates space for dense Vector

		cudaMalloc(v_gpuPtr, n*Sizeof.DOUBLE);
	}

	public double[] getVector(){
		// returns double array on host
		double v_host[] = new double[size];

		cudaMemcpy(Pointer.to(v_host), v_gpuPtr, size*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

		return v_host;
	}

	public Pointer getPtr(){
		// returns gpu pointer
		return v_gpuPtr;
	}

	public void free(){
		// frees gpu memory
		cudaFree(v_gpuPtr);
	}

}


