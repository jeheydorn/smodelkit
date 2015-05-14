package smodelkit.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import org.junit.Test;

import smodelkit.util.Tuple2;
import smodelkit.util.Tuple2Iterator;

public class Tuple2IteratorTest
{

	@Test
	public void test()
	{
		List<Integer> it1 = Arrays.asList(1, 2, 3);
		List<Integer> it2 = Arrays.asList(4, 5, 6);
		Tuple2Iterator<Integer, Integer> target = new Tuple2Iterator<Integer, Integer>(it1, it2);
		assertTrue(target.hasNext());
		assertEquals(new Tuple2<>(1, 4), target.next());
		assertTrue(target.hasNext());
		assertEquals(new Tuple2<>(2, 5), target.next());
		assertTrue(target.hasNext());
		assertEquals(new Tuple2<>(3, 6), target.next());
		assertFalse(target.hasNext());	
	}

	@Test
	public void differentSizeIterablesTest()
	{
		List<Integer> it1 = Arrays.asList(1, 2);
		List<Integer> it2 = Arrays.asList(4, 5, 6);
		Tuple2Iterator<Integer, Integer> target = new Tuple2Iterator<Integer, Integer>(it1, it2);
		assertTrue(target.hasNext());
		assertEquals(new Tuple2<>(1, 4), target.next());
		assertTrue(target.hasNext());
		assertEquals(new Tuple2<>(2, 5), target.next());
		assertTrue(target.hasNext());
		try
		{
			target.next();
		}
		catch (NoSuchElementException e)
		{
			// success
			return;
		}
		
		assertTrue(false);
	}

	@Test
	public void oneEmptyIterableTest()
	{
		List<Integer> it1 = Arrays.asList(1, 2);
		List<Integer> it2 = Arrays.asList();
		Tuple2Iterator<Integer, Integer> target = new Tuple2Iterator<Integer, Integer>(it1, it2);
		assertTrue(target.hasNext());
		
		try
		{
			target.next();
		}
		catch (NoSuchElementException e)
		{
			// success
			return;
		}
		
		assertTrue(false);
	}
	
	@Test
	public void bothEmptyIterableTest()
	{
		List<Integer> it1 = Arrays.asList();
		List<Integer> it2 = Arrays.asList();
		Tuple2Iterator<Integer, Integer> target = new Tuple2Iterator<Integer, Integer>(it1, it2);
		assertFalse(target.hasNext());
	}


}
