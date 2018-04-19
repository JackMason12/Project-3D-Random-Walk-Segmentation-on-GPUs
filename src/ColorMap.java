import java.awt.Color;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class ColorMap {
	/*
	 * Class used to generate colours for mask
	 * first 3 colours are r,g,b
	 * any subsequent colours are randomised
	 * we allocate 1 colour per label
	 */
	
	public int[] map;
	
	public ColorMap(int label_count) {
		map = new int[label_count];
		int r = 0;
		int g = 0;
		int b = 0;
		int rgb = 0;
		ArrayList<Integer> prevs = new ArrayList<Integer>();
		//Color.GREEN
		//Color.RED
		//Color.BLUE
		//Color.YELLOW
		for (int i = 0; i < label_count; i++) {
			if (i < 3) {
				if (i == 0) {
					rgb = Color.RED.getRGB();
				} else if (i == 1) {
					rgb = Color.GREEN.getRGB();
				} else if (i == 2) {
					rgb = Color.BLUE.getRGB();
				}
			}
			if (i >= 3) {
				while (prevs.contains(rgb)) { //generate a new colour that we havent used before
					//probably going to be very inefficient if you want like 10million+ labels (please dont)
					r = ThreadLocalRandom.current().nextInt(0, 256);
					g = ThreadLocalRandom.current().nextInt(0, 256);
					b = ThreadLocalRandom.current().nextInt(0, 256);
					rgb = new Color(r, g, b).getRGB();
				}
			}
			map[i] = rgb;
			prevs.add(rgb);
		}
	}
}
