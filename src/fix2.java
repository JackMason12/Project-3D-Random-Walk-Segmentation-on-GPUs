import java.awt.Color;

public class fix2 {
	public static double[][] getProbs(int[] pixels, Edge[] edges, double beta, int[] seeds, int[] labels, int labelCount) {
		
		
		final int pixelCount = pixels.length; //constants
		final int edgeCount = edges.length;
		final int seedCount = seeds.length;
		
		/*
		 * Matrix A is a edgexpixel matrix
		 * For each row (edge), there is an entry at each of the pixels(columns)
		 * corresponding to an end of the edge, assigned an arbitrary orientation (one value is 1 and the other is -1)
		 */
		
		final int nnzA = edgeCount*2; //2 entries for each edge, therefore 2*edgecount non zero entries
		
		int ARow[] = new int[nnzA]; //arrays to hold matrix a in coo format
		int ACol[] = new int[nnzA];
		double AVal[] = new double[nnzA]; //use doubles for all values
		
		int AEntries = 0;
		for (int i = 0; i < edgeCount; i++) {
			ARow[AEntries] = edges[i].start;
			ACol[AEntries] = i;
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		return null;
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
