package smodelkit.test;

import static org.junit.Assert.*;
import static smodelkit.Vector.assertVectorEquals;

import org.junit.Assert;
import org.junit.Test;

import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.util.Range;

public class VectorTest
{

	@Test
	public void constructorTest()
	{
		{
			double[] values = new double[]{1.0, 2.0, 3.0};
			Vector actual = Vector.create(values);
			assertEquals(1.0, actual.getWeight(), 0);
			for (int i : new Range(values.length))
			{
				assertEquals(values[i], actual.get(i), 0.0);
			}
		}

		{
			double[] values = new double[]{1.0, 2.0, 3.0};
			Vector actual = Vector.create(values, 2.0);
			assertEquals(2.0, actual.getWeight(), 0);
			for (int i : new Range(values.length))
			{
				assertEquals(values[i], actual.get(i), 0.0);
			}
		}

		{
			double[] values = new double[]{1.0, 2.0, 3.0};
			Vector actual = Vector.create(Vector.create(values), 2.0);
			assertEquals(2.0, actual.getWeight(), 0);
			for (int i : new Range(values.length))
			{
				assertEquals(values[i], actual.get(i), 0.0);
			}
		}

		{
			double[] values = new double[]{1.0, 2.0, 3.0};
			Vector actual = Vector.create(Vector.create(values));
			assertEquals(1.0, actual.getWeight(), 0);
			for (int i : new Range(values.length))
			{
				assertEquals(values[i], actual.get(i), 0.0);
			}
		}
	}
	
	@Test
	public void setTest()
	{
		double[] values = new double[]{1.0, 2.0, 3.0};
		Vector actual = Vector.create(values);
		assertEquals(1.0, actual.getWeight(), 0);
		actual.set(1, 10);
		assertEquals(10, actual.get(1), 0);
		assertEquals(3, actual.size());
		// Make sure the original vector was not changed.
		assertEquals(2, values[1], 0);
	}
	
	@Test
	public void subVectorTest()
	{
		subVectorTest(new double[]{1.0, 2.0, 3.0, 4.0}, 1, 3);
		subVectorTest(new double[]{1.0, 2.0, 3.0, 4.0}, 0, 4);
		subVectorTest(new double[]{1.0, 2.0, 3.0, 4.0}, 0, 1);
		subVectorTest(new double[]{1.0, 2.0, 3.0, 4.0}, 3, 4);

		subVectorTest(new double[]{8, 4, 6, -1}, 1, 3);
		subVectorTest(new double[]{8, 4, 6, -1}, 0, 4);
		subVectorTest(new double[]{8, 4, 6, -1}, 0, 1);
		subVectorTest(new double[]{8, 4, 6, -1}, 3, 4);
	}
	
	private void subVectorTest(double[] values, int from, int to)
	{
		Vector v = Vector.create(values, 2.0);
		Vector actual = v.subVector(from, to);
		assertEquals(2.0, actual.getWeight(), 0);
		assertEquals(to - from, actual.size());
		for (int i : new Range(to - from))
		{
			assertEquals(values[i + from], actual.get(i), 0);
		}
		
		// Test creating a sub-vector from a sub-vector.
		Vector subActual = actual.subVector(0, actual.size());
		assertVectorEquals(actual, subActual, 0);
		assertEquals(2.0, subActual.getWeight(), 0);
		if (to - from > 1)
		{
			// Increase "from" by 1.
			subActual = actual.subVector(1, actual.size());
			assertEquals(actual.size() - 1, subActual.size());
			for (int i : new Range(subActual.size()))
			{
				assertEquals(actual.get(i + 1), subActual.get(i), 0);
			}
			
			// Decrease "to" by 1.
			subActual = actual.subVector(0, actual.size() - 1);
			assertEquals(actual.size() - 1, subActual.size());
			for (int i : new Range(subActual.size()))
			{
				assertEquals(actual.get(i), subActual.get(i), 0);
			}
		}
	}
		
	@Test(expected=IllegalArgumentException.class)
	public void subVectorNegativeTest1()
	{
		Vector.create(new double[]{1.0, 2.0, 3.0, 4.0}).subVector(-1, 2);
	}

	@Test(expected=IllegalArgumentException.class)
	public void subVectorNegativeTest2()
	{
		Vector.create(new double[]{1.0, 2.0, 3.0, 4.0}).subVector(1, 5);
	}
	
	@Test
	public void addAllTest()
	{
		Vector v = Vector.create(1.0, 2.0, 3.0);
		v.addAll(Vector.create(4.0));
		assertVectorEquals(Vector.create(1, 2, 3, 4), v, 0);
	}
	
	@Test public void assertVectorEqualsTest1()
	{
		try
		{
			assertVectorEquals(Vector.create(1, 2, 3), Vector.create(1, 2), 0);
		}
		catch(AssertionError e)
		{
			// success
			return;
		}
		Assert.fail();
	}

	@Test public void assertVectorEqualsTest2()
	{
		try
		{
			assertVectorEquals(Vector.create(1, 2), Vector.create(1, 2, 3), 0);
		}
		catch(AssertionError e)
		{
			// success
			return;
		}
		Assert.fail();
	}

	@Test public void assertVectorEqualsTest3()
	{
		try
		{
			assertVectorEquals(Vector.create(new double[]{1, 2}, 2.0), Vector.create(1, 2), 0);
		}
		catch(AssertionError e)
		{
			// success
			return;
		}
		Assert.fail();
	}

	@Test public void assertVectorEqualsTest4()
	{
		try
		{
			assertVectorEquals(Vector.create(1, 2), Vector.create(1, 3), 0);
		}
		catch(AssertionError e)
		{
			// success
			return;
		}
		Assert.fail();
	}
	
	@Test
	public void concatTest1()
	{
		Vector expected = Vector.create(1, 2);
		assertVectorEquals(expected, Vector.create(1).concat(Vector.create(2)), 0);		
	}
	@Test
	public void concatTest2()
	{
		assertVectorEquals(Vector.create(1, 2), Vector.create(1).concat(new double[] {2}), 0);		
	}
	
	@Test
	public void removeTest()
	{
		double[] values = new double[]{1.0, 2.0, 3.0};
		Vector actual = Vector.create(values);
		actual.remove(1);
		assertEquals(1, actual.get(0), 0);
		assertEquals(3, actual.get(1), 0);
		assertEquals(2, actual.size());
		// Make sure the original vector was not changed.
		assertArrayEquals(new double[]{1.0, 2.0, 3.0}, values, 0);
	}
}
