package smodelkit.util;

import java.util.Comparator;

/**
 * For comparing double arrays of the same length.
 * @author joseph
 *
 */
public class DoubleArrayComparator implements Comparator<double[]>
{

	@Override
	public int compare(double[] d1, double[] d2)
	{
		assert d1.length == d2.length;
		for (int i = 0; i < d1.length; i++)
		{				
			if (d1[i] < d2[i])
				return -1;
			if (d1[i] > d2[i])
				return 1;
		}
		return 0;
	}
}
