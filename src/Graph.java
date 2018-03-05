import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

public class Graph {
	
	public ArrayList<Edge> edges;
	public ArrayList<Pixel> pixels;
	private double beta;
	
	public Graph(BufferedImage image, double beta) {
		this.edges = new ArrayList<Edge>();
		this.pixels = new ArrayList<Pixel>();
		this.beta = beta;
		int width = image.getWidth();
		int height = image.getHeight();
		for (int i = 0; i < width; i++) { //for every pixel attempt to add an edge if it is not already present (4 neighbours)
			for (int j = 0; j < height; j++) {
				System.out.println(String.format("(%2d, %2d)", i, j));
				Pixel pixela = new Pixel(image.getRGB(i, j), i, j);
				Pixel pixelb;
				if (i-1 >= 0) {
					pixelb = new Pixel(image.getRGB(i-1, j), i-1, j);
					this.AttemptAdd(pixela, pixelb);
				}
				if (j-1 >= 0) {
					pixelb = new Pixel(image.getRGB(i,  j-1), i, j-1);
					this.AttemptAdd(pixela, pixelb);
				}
				if (i+1 < width) {
					pixelb = new Pixel(image.getRGB(i+1, j), i+1, j);
					this.AttemptAdd(pixela, pixelb);
				}
				if (j+1 < height) {
					pixelb = new Pixel(image.getRGB(i, j+1), i, j+1);
					this.AttemptAdd(pixela, pixelb);
				}
				
			}
		}
	}
	
	private void AttemptAdd(Pixel pixela, Pixel pixelb) {
		if (!pixels.contains(pixela)) pixels.add(pixela);
		if (!pixels.contains(pixelb)) pixels.add(pixelb);
		Edge toAdd = new Edge(pixela, pixelb, this.beta);
		if (!edges.contains(toAdd) && !edges.contains(toAdd.Inverse())) {
			edges.add(toAdd);
		}
	}
	
	private class Edge {
		private Pixel a;
		private Pixel b;
		private double weight;
		private double beta;
		
		public Edge(Pixel a, Pixel b, double beta) {
			this.a = a;
			this.b = b;
			this.beta = beta;
			this.setWeight();
		}
		
		public Edge Inverse()  {
			return new Edge(this.b, this.a, this.beta); //return an inverse of this edge
		}
		
		private void setWeight() {
			//some way of finding weight using a and b
			int reddiff = this.a.getRed() - this.b.getRed();
			int greendiff = this.a.getGreen() - this.b.getGreen();
			int bluediff = this.a.getBlue() - this.b.getBlue();
			int alphadiff = this.a.getAlpha() - this.b.getAlpha();
			double g = Math.sqrt(Math.pow(reddiff, 2) + Math.pow(greendiff,  2) + Math.pow(bluediff, 2) + Math.pow(alphadiff, 2));
			this.weight = Math.exp(-(this.beta * Math.pow(g, 2))); //standard gaussian weighting function
		}
		
		public Pixel getA() {
			return this.a;
		}
		
		public Pixel getB() {
			return this.b;
		}
		
		public double getWeight() {
			return this.weight;
		}
		
		public boolean equals(Edge e) { //returns true if given edge is equal to this one
			//if the pair of pixels are the same (either order) and the weight of the edge is the same
			if (this.a.equals(e.getA()) && this.b.equals(e.getB())) {
				if (this.weight == e.getWeight()) return true;
			}
			if (this.b.equals(e.getA()) && this.a.equals(e.getB())) {
				if (this.weight == e.getWeight()) return true;
			}
			return false;
		}
		
		public String toString() {
			return String.format("Pixels at (%2d, %2d) (%2d, %2d) weight %f", this.a.getX(), this.a.getY(), this.b.getX(), this.b.getY(), this.weight);
		}
		
	}
	
	private class Pixel {
		private Color data;
		private int x;
		private int y;
		
		public Pixel(int RGB, int x, int y) {
			this.data = new Color(RGB);
			this.x = x;
			this.y = y;
		}
		
		public int getX() {
			return this.x;
		}
		
		public int getY() {
			return this.y;
		}
		
		public Color getData() {
			return this.data;
		}
		
		public int getRed() {
			return this.data.getRed();
		}
		
		public int getGreen() {
			return this.data.getGreen();
		}
		
		public int getBlue() {
			return this.data.getBlue();
		}
		
		public int getAlpha() {
			return this.data.getAlpha();
		}
		
		public boolean equals(Pixel p) {
			if ((this.x == p.getX()) && (this.y == p.getY()) && this.data.equals(p.getData())) return true;
			return false;
		}
		
		public String toString() {
			return String.format("Red: %2d, Green: %2d, Blue: %2d, Alpha: %2d at (%2d, %2d)", this.getRed(), this.getGreen(), this.getBlue(), this.getAlpha(), this.x, this.y);
		}
		
	}

}
