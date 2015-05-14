package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.util.ArithmeticUtils;
import org.junit.Test;

import smodelkit.util.CombinationIterator;

@SuppressWarnings("unused")
public class CombinationIteratorTest
{

	@Test
	public void testThree()
	{
		int choose = 3;
		List<String> items = Arrays.asList(new String[] {"a", "b", "c", "d", "e"});
		CombinationIterator<String> iter = new CombinationIterator<String>(items, choose);
		long count = 0;
		for (List<String> strList : iter)
		{
			count++;
		}	
		assertEquals(ArithmeticUtils.binomialCoefficient(items.size(), choose), count);
	}
	
	@Test
	public void testTwo()
	{
		int choose = 2;
		List<String> items = Arrays.asList(new String[] {"a", "b", "c", "d", "e", "f"});
		CombinationIterator<String> iter = new CombinationIterator<String>(items, choose);
		long count = 0;
		for (List<String> strList : iter)
		{
			count++;
		}	
		assertEquals(ArithmeticUtils.binomialCoefficient(items.size(), choose), count);
	}

	
	@Test
	public void testMany()
	{
		int choose = 5;
		List<String> items = new ArrayList<String>();
		for (int i = 0; i < 30; i++)
		{
			items.add(String.format("%s", i));
		}
		long count = 0;
		for (List<String> strList :  new CombinationIterator<String>(items, choose))
		{
			count++;
		}	
		assertEquals(ArithmeticUtils.binomialCoefficient(items.size(), choose), count);
	}

	@Test
	public void testOne()
	{
		int choose = 1;
		List<String> items = new ArrayList<String>();
		for (int i = 0; i < 30; i++)
		{
			items.add(String.format("%s", i));
		}
		CombinationIterator<String> iter = new CombinationIterator<String>(items, choose);
		long count = 0;
		for (List<String> strList : iter)
		{
			count++;
		}	
		assertEquals(ArithmeticUtils.binomialCoefficient(items.size(), choose), count);
	}
	
	@Test
	public void testChooseAll()
	{
		int choose = 30;
		List<String> items = new ArrayList<String>();
		for (int i = 0; i < 30; i++)
		{
			items.add(String.format("%s", i));
		}
		CombinationIterator<String> iter = new CombinationIterator<String>(items, choose);
		long count = 0;
		for (List<String> strList : iter)
		{
			count++;
		}	
		assertEquals(ArithmeticUtils.binomialCoefficient(items.size(), choose), count);
	}



}
