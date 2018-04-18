
public class UnitTest {
	public static boolean TestDenseTranspose() {
		double[][] inData = new double[][] {{1.0, 1.0, 1.0, 1.0, 1.0},
										{2.0, 2.0, 2.0, 2.0, 2.0},
										{3.0, 3.0, 3.0, 3.0, 3.0},
										{4.0, 4.0, 4.0, 4.0, 4.0}};
		int inM = 4;
		int inN = 5;
		
		double[][] expectedOut = new double[][] {{1.0,2.0,3.0,4.0},
												 {1.0,2.0,3.0,4.0},
												 {1.0,2.0,3.0,4.0},
												 {1.0,2.0,3.0,4.0},
												 {1.0,2.0,3.0,4.0}};
		int expectedM = 5;
		int expectedN = 4;
		
		DenseMatrix in = new DenseMatrix(inM, inN, inData);
		DenseMatrix T = in.Transpose();
		
		
		boolean mSame = false;
		boolean nSame = false;
		boolean dataSame = false;
		
		if (T.m == expectedM) {
			mSame = true; 
		} else {
			System.out.println("M different");
			return false;
		}
		if (T.n == expectedN) {
			nSame = true;
		} else {
			System.out.println("N different");
			return false;
		}
		for (int i = 0; i < expectedM; i++) {
			for (int j = 0; j < expectedN; j++) {
				if (expectedOut[i][j] != T.data[i][j]) {
					dataSame = false;
				}
			}
		}
		
		if (mSame && nSame && dataSame) return true;
		return false;
		
	}
}
