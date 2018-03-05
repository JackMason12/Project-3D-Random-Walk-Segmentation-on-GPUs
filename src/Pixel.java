import java.awt.Color;

public class Pixel {
	
	public Color data;
	public int id;
	
	public Pixel(int data, int id) {
		this.data = new Color(data);
		this.id = id;		
	}
	
	public Pixel(Pixel p) {
		this.data = p.data;
		this.id = p.id;
	}
	
	public boolean equals(Pixel p) {
		if ((this.data == p.data) && (this.id == p.id)) return true;
		return false;
	}
	
}
