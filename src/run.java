
public class run {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int pixel[] = new int[] {Integer.MAX_VALUE,Integer.MAX_VALUE,Integer.MAX_VALUE,Integer.MAX_VALUE,Integer.MAX_VALUE,Integer.MAX_VALUE};
		int pixelCount = 6;
		
		Edge edge[] = new Edge[5];
		edge[0] = new Edge(0, 1);
		edge[1] = new Edge(1, 2);
		edge[2] = new Edge(2, 3);
		edge[3] = new Edge(3, 4);
		edge[4] = new Edge(4, 5);
		int edgeCount = 5;
		
		double beta = 0.1;
		
		int[] seeds = new int[]{0,5};
		int[] labels = new int[]{0,1};
		GPURW rw = new GPURW();
		
		rw.InitFromArrays(pixel, edge);
		DenseMatrix results = rw.GetProbabilities(seeds, labels, 2);
		//System.out.println(results.toString());
	}

}
