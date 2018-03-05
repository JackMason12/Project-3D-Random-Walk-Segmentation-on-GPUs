import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import javax.imageio.ImageIO;

public class Test {

	public static void main(String[] args){
		// TODO Auto-generated method stub
		BufferedImage image = null;
		System.out.print("Enter image path: ");
		try { //read in image
			Scanner scanner = new Scanner(System.in);
			String imgaddress = scanner.nextLine();
			image = ImageIO.read(new File(imgaddress));
			scanner.close();
		
		} catch (Exception e) {
			
		}
		System.out.print("\n");
		int width = image.getWidth(); //get image width
		int height = image.getHeight(); //get image height
		int pixelCount = width*height; //compute number of pixels
	    int edgeCount = (width*(height-1))+(height*(width-1)); //compute number of edges
	    int pixel[] = new int[pixelCount]; //array of pixels, numbered l-r t-b
	    Edge edge[] = new Edge[edgeCount]; //array of edges, numbered horizontally l-r followed by vertically t-b
	    double beta = (double) 1.5;
	    
	    int labels[] = new int[] {0,0,0,1,1,0,0,1,1,1};
	    int seeds[] = new int[] {0, 1, 10, 3, 4, 11, 20, 5, 6, 7};
	    
	    for (int j = 0; j < width; j++) {
	      for (int i = 0; i < height; i++) {
	        pixel[j*height+i] = image.getRGB(i, j); //read in each pixel (l-r, t-b);
	      }
	    }
	    
	    
	    int edgeIndex = 0;//read in each edge, left-right
	    //horizontal edges
	    for (int j = 0; j < height; j++) {
	      for (int i = 0; i < width; i++) {
	        if (i != width-1) {
	        	edge[edgeIndex++] = new Edge((j*height + i), (j*height + i + 1));
	        }
	      }
	    }
	    //vertical edges
	    for (int i = 0; i < width; i++) { //read in each edge, top-bottom
	      for (int j = 0; j < height; j++) {
	        if (j != height-1) edge[edgeIndex++] = new Edge(j*height + i, (j+1)*height + i);
	      }
	    }
		double proba[][] = RandomWalkSegmentationGPU.getProbabilities(pixel, pixelCount, edge, edgeCount, beta, seeds, labels);

	}

}
