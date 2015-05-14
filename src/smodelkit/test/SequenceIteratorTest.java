package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import smodelkit.util.Helper;
import smodelkit.util.SequenceIterator;

public class SequenceIteratorTest
{

	@Test
	public void test1()
	{
		int count = 0;
		for (@SuppressWarnings("unused") List<Integer> seq : new SequenceIterator(Arrays.asList(2)))
		{
			count++;
		}
		
		assertEquals(2, count);
	}

	@Test (expected = IllegalArgumentException.class)
	public void test2()
	{
		int count = 0;
		for (@SuppressWarnings("unused") List<Integer> seq : new SequenceIterator(Arrays.asList(0)))
		{
			count++;
		}
		
		assertEquals(0, count);
	}

	@Test
	public void test3()
	{
		int count = 0;
		for (@SuppressWarnings("unused") List<Integer> seq : new SequenceIterator(Arrays.asList(1)))
		{
			count++;
		}
		
		assertEquals(1, count);
	}
	
	@Test
	public void test4()
	{
		Iterator<List<Integer>> it = new SequenceIterator(Arrays.asList(2, 3));
		
		List<List<Integer>> actual = Helper.iteratorToList(it);
		
		List<List<Integer>> expected = Arrays.asList(
				Arrays.asList(0,0),
				Arrays.asList(0,1),
				Arrays.asList(0,2),
				Arrays.asList(1,0),
				Arrays.asList(1,1),
				Arrays.asList(1,2));

		
		assertEquals(expected, actual);
	}


	@Test
	public void test5()
	{
		Iterator<List<Integer>> it = new SequenceIterator(Arrays.asList(3, 2, 3));
		
		List<List<Integer>> actual = Helper.iteratorToList(it);
		
		List<List<Integer>> expected = Arrays.asList(
				Arrays.asList(0,0,0),
				Arrays.asList(0,0,1),
				Arrays.asList(0,0,2),
				Arrays.asList(0,1,0),
				Arrays.asList(0,1,1),
				Arrays.asList(0,1,2),
				Arrays.asList(1,0,0),
				Arrays.asList(1,0,1),
				Arrays.asList(1,0,2),
				Arrays.asList(1,1,0),
				Arrays.asList(1,1,1),
				Arrays.asList(1,1,2),
				Arrays.asList(2,0,0),
				Arrays.asList(2,0,1),
				Arrays.asList(2,0,2),
				Arrays.asList(2,1,0),
				Arrays.asList(2,1,1),
				Arrays.asList(2,1,2));

		
		assertEquals(expected, actual);
	}

	
}
