import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class Test {

	public static void main(String[] args) throws IOException{
		BufferedImage image = null;
		System.out.print("Enter image path: ");
		Scanner scanner = new Scanner(System.in);
		try { //read in image
			//Scanner scanner = new Scanner(System.in);
			String imgaddress = scanner.nextLine();
			image = ImageIO.read(new File(imgaddress));
			//scanner.close();
		
		} catch (Exception e) {
			
		}
		System.out.println();
		String in = "empty";
		int curLabel = 0;
		String[] temp_seeds = {"hi"};
		ArrayList<Integer> labs = new ArrayList<Integer>();
		ArrayList<Integer> sds = new ArrayList<Integer>();
		boolean loop = true;
		//Scanner scanner = new Scanner(System.in);
		while (loop) {
			System.out.println("Enter comma seperated seeds for label " + curLabel + ": ");
			try {
				//Scanner scanner = new Scanner(System.in);
				in = scanner.nextLine();
				if (in.length() == 0) loop = false;
				temp_seeds = new String[in.split(",").length];
				temp_seeds = in.split(",");
				for (String sd : temp_seeds) {
					labs.add(curLabel);
					sds.add(Integer.parseInt(sd));
				}
				curLabel++;
			} catch (Exception e) {
				
			}
			System.out.println(temp_seeds.length);
		}
		scanner.close();
		//System.out.print("\n");
		for (int i = 0; i < image.getHeight(); i++) {
			for (int j = 0; j < image.getWidth(); j++) {
				Color col = new Color(image.getRGB(j, i));
				//System.out.print(image.getRGB(j, i));
				if (j != image.getWidth()-1) {
					//System.out.print(",");
				}
			}
			//System.out.print("\n");
		}
		int width = image.getWidth(); //get image width
		int height = image.getHeight(); //get image height
		int pixelCount = width*height; //compute number of pixels
		int edgeCount = (width*(height-1))+(height*(width-1)); //compute number of edges
	    int pixel[] = new int[pixelCount]; //array of pixels, numbered l-r t-b
	    Edge edge[] = new Edge[edgeCount]; //array of edges, numbered horizontally l-r followed by vertically t-b
	    double beta =  1.0;
	    
	    
	    
	    //int labels[] = new int[] {0,1};
	    //int label_count = 2;
	    //int seeds[] = new int[] {9900, 99};
	    Integer[] labels_temp = labs.toArray(new Integer[labs.size()]);
	    int[] labels = new int[labels_temp.length];
	    for (int i = 0; i < labels_temp.length; i++) {
	    	labels[i] = labels_temp[i];
	    }
	    Integer[] seeds_temp = sds.toArray(new Integer[sds.size()]);
	    int[] seeds = new int[seeds_temp.length];
	    for (int i = 0; i < seeds_temp.length; i++) {
	    	seeds[i] = seeds_temp[i];
	    }
	    int label_count = curLabel;
	    
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
		/*
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
		
		
		
		double proba[][] = RandomWalkSegmentationGPU.getProbabilities(pixel, pixelCount, edge, edgeCount, beta, seeds, labels);*/
		double proba[][] = RandomWalkSegmentationGPU.getProbabilities(pixel, pixelCount, edge, edgeCount, beta, seeds, labels);
		double probs[][] = MatrixUtils.AddSeeds(proba, seeds, labels);
		int[][] mask = MatrixUtils.GetMask(probs, height, width, label_count);
		
		File originalFile = new File("bin/original.jpg");
		ImageIO.write(image, "jpg", originalFile);
		BufferedImage maskImage = MatrixUtils.GetMaskImage(mask, label_count);
		BufferedImage seedImage = MatrixUtils.GetSeedImage(image, seeds, labels);
		File maskFile = new File("bin/mask.jpg");
		File seedFile = new File("bin/seed.jpg");
		ImageIO.write(maskImage, "jpg", maskFile);
		ImageIO.write(seedImage, "jpg", seedFile);
		
		JFrame maskFrame = new JFrame();
		maskFrame.add(new JPanel().add(new JLabel(new ImageIcon(maskImage))));
		maskFrame.setTitle("Mask");
		maskFrame.setSize(1280, 720);
		maskFrame.setVisible(true);
		maskFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		JFrame originalFrame = new JFrame();
		originalFrame.add(new JPanel().add(new JLabel(new ImageIcon(image))));
		originalFrame.setTitle("Original");
		originalFrame.setSize(1280, 720);
		originalFrame.setVisible(true);
		originalFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		JFrame seedFrame = new JFrame();
		seedFrame.add(new JPanel().add(new JLabel(new ImageIcon(seedImage))));
		seedFrame.setTitle("Seeds");
		seedFrame.setSize(1280, 720);
		seedFrame.setVisible(true);
		seedFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

}
