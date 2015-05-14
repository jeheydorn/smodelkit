package smodelkit.test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.junit.Test;

import smodelkit.util.Helper;
import smodelkit.util.Range;


public class HelperTest
{

	@Test
	public void maxElementTest()
	{
		double[] d = {0.1, 0.2, -7, 9, 8};
		assertEquals(9, Helper.maxElement(d), 0.0);

		d = new double[]{10, 0.2, -7, 9, 8};
		assertEquals(10, Helper.maxElement(d), 0.0);
	}
	
	@Test
	public void indexOfMaxElementInRangeTest()
	{
		double[] d = {0.1, 0.2, -7, 9, 8};
		assertEquals(3, Helper.indexOfMaxElementInRange(d, 0, 5));

		d = new double[]{10, 0.2, -7, 9, 8};
		assertEquals(0, Helper.indexOfMaxElementInRange(d, 0, 5));

		d = new double[]{10, 0.2, -7, 9, 8};
		assertEquals(3, Helper.indexOfMaxElementInRange(d, 1, 4));

		d = new double[]{-20, 0.2, -7, 9, 8};
		assertEquals(0, Helper.indexOfMaxElementInRange(d, 0, 1));
	}
	
	@Test
	public void indexOfMaxElementTest()
	{
		// Without comparator.
		{
			List<Double> d = Arrays.asList(0.1, 0.2, -7.0, 9.0, 8.0);
			assertEquals(3, Helper.indexOfMaxElement(d));
		}
		{
			List<Double> d = Arrays.asList(0.1, 0.2, -7.0, 9.0, 10.0);
			assertEquals(4, Helper.indexOfMaxElement(d));
		}
		{
			List<Double> d = Arrays.asList(10.0, 0.2, -7.0, 9.0, 8.0);
			assertEquals(0, Helper.indexOfMaxElement(d));
		}	
		{
			List<Double> d = Arrays.asList(10.0, 20.0, -7.0, 9.0, 8.0);
			assertEquals(1, Helper.indexOfMaxElement(d));
		}
		{
			List<Double> d = Arrays.asList(10.0);
			assertEquals(0, Helper.indexOfMaxElement(d));
		}
		
		// With comparator.
		Comparator<Double> reverse = new Comparator<Double>()
			{
				public int compare(Double d1, Double d2)
				{
					return -Double.compare(d1, d2);
				}
			};
		{
			List<Double> d = Arrays.asList(-0.1, -0.2, 7.0, -9.0, -8.0);
			assertEquals(3, Helper.indexOfMaxElement(d, reverse));
		}
		{
			List<Double> d = Arrays.asList(-0.1, -0.2, 7.0, -9.0, -10.0);
			assertEquals(4, Helper.indexOfMaxElement(d, reverse));
		}
		{
			List<Double> d = Arrays.asList(-10.0, -0.2, 7.0, -9.0, -8.0);
			assertEquals(0, Helper.indexOfMaxElement(d, reverse));
		}	
		{
			List<Double> d = Arrays.asList(-10.0, -20.0, 7.0, -9.0, -8.0);
			assertEquals(1, Helper.indexOfMaxElement(d, reverse));
		}
		{
			List<Double> d = Arrays.asList(10.0);
			assertEquals(0, Helper.indexOfMaxElement(d, reverse));
		}
		
	}
	
	@Test
	public void concatArraysTest()
	{
		double[] d1 = {0.1, 0.2, -7, 9, 8};
		double[] d2 = {-100000, 0, 6};
		double[] expected = {0.1, 0.2, -7, 9, 8, -100000, 0, 6};
		assertArrayEquals(expected, Helper.concatArrays(d1, d2), 0);
	}
	
	@Test
	public void testSortIndexes()
	{
		double[] vals = new double[] {0.1, 0.2, 0.3, 7, -500};
		Integer[] actual = Helper.sortIndexesDescending(vals);
		assertArrayEquals(new Integer[] {3,  2, 1, 0, 4}, actual);
	}
	
	@Test
	public void findVarianceTest()
	{
		{
			List<Double> values = Arrays.asList(600.0, 470.0, 170.0, 430.0, 300.0);
			assertEquals(21704, Helper.findVariance(values), 0.000000001);
		}

		{
			List<Double> values = Arrays.asList(-600.0, -470.0, 170.0, -430.0, 300.0);
			// mean = −206.
			// 155236 + 69696 + 141376 + 50176 + 256036
			assertEquals(134504, Helper.findVariance(values), 0.000000001);
		}
}
	
	@Test
	public void findStandardDeviationTest()
	{
		{
			List<Double> values = Arrays.asList(600.0, 470.0, 170.0, 430.0, 300.0);
			assertEquals(147.322774886, Helper.findStandardDeviation(values), 0.000000001);
		}

		{
			List<Double> values = Arrays.asList(-600.0, -470.0, 170.0, -430.0, 300.0);
			assertEquals(366.747869796, Helper.findStandardDeviation(values), 0.000000001);
		}

		{
			List<Double> values = Arrays.asList(0.0);
			assertEquals(0.0, Helper.findStandardDeviation(values), 0.000000001);
		}
	}
	
	@Test
	public void deepCopyTest()
	{
		List<Integer> expected = Arrays.asList(1, 2, 3, 4);
		@SuppressWarnings("unchecked")
		List<Integer> actual = (List<Integer>)Helper.deepCopy(expected);
		assertEquals(expected, actual);
		assertFalse(expected == actual);
	}

	@Test
	public void deepCopyDeepTest()
	{
		List<List<Integer>> expected = Arrays.asList(Arrays.asList(1, 2, 3), 
				Arrays.asList(4, 5, 6),
				Arrays.asList(7, 8, 9),
				Arrays.asList(10, 11, 12));
		@SuppressWarnings("unchecked")
		List<List<Integer>> actual = (List<List<Integer>>)Helper.deepCopy(expected);
		assertEquals(expected, actual);
		assertFalse(expected == actual);
		for (int i : new Range(expected.size()))
		{
			assertEquals(expected.get(i), actual.get(i));
			assertFalse(expected.get(i) == actual.get(i));
		}
	}


}
