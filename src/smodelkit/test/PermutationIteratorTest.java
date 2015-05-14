package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.util.ArithmeticUtils;
import org.junit.Test;

import smodelkit.util.PermutationIterator;

public class PermutationIteratorTest
{
	@Test
	public void test()
	{
		List<String> items = Arrays.asList(new String[] {"a", "b", "c", "d"});
		int count = 0;
		PermutationIterator<String> iter = new PermutationIterator<String>(items);
		while(iter.hasNext())
		{
			iter.next();
			//System.out.println(iter.next());
			count++;
		}
		assertEquals(ArithmeticUtils.factorial(items.size()), count);
	}

	@Test
	public void testMany()
	{
		for (int n = 1; n < 8; n++)
		{
			List<String> items = Collections.nCopies(n, "");
			int count = 0;
			PermutationIterator<String> iter = new PermutationIterator<String>(items);
			while(iter.hasNext())
			{
				iter.next();
				count++;
			}
			assertEquals(ArithmeticUtils.factorial(items.size()), count);
		}
	}

}
