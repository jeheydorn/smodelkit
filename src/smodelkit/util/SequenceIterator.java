package smodelkit.util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Iterates through all possible fixed length sequences where each element
 * comes from a specified domain.
 * @author joseph
 *
 */
public class SequenceIterator implements Iterable<List<Integer>>, Iterator<List<Integer>> 
{
	/**
	 * Stores the number of sequences that have been returned so far.
	 */
	private long n;
	private List<Integer> domainSizes;
	
	/**
	 * 
	 * @param domainSizes Each element specifies the number of values an element of a
	 * returned sequence can take on. 
	 */
	public SequenceIterator(List<Integer> domainSizes)
	{
		if (domainSizes.isEmpty())
			throw new IllegalArgumentException();
		for (int i : new Range(domainSizes.size()))
		{
			if (domainSizes.get(i) == 0)
				throw new IllegalArgumentException("A domain cannot be size zero.");
		}
		n = 0;
		this.domainSizes = domainSizes;
	}

	@Override
	public boolean hasNext()
	{
		
		return n < calcNumPossibleUniqueSequences(domainSizes);
	}
	
	private static long calcNumPossibleUniqueSequences(List<Integer> domainSizes)
	{
		int product = 1;
		for (int i : new Range(domainSizes.size()))
		{
			product *= domainSizes.get(i);
		}		
		return product;
	}

	@Override
	public List<Integer> next()
	{
		List<Integer> result = calcNthSequence(domainSizes, n);
		n++;
		return result;
	}
	
	private static List<Integer> calcNthSequence(List<Integer> domainSizes, long n)
	{
		List<Integer> result = new ArrayList<>(domainSizes.size());
		long remainder = n;
		for (int i : new Range(domainSizes.size()))
		{
			int product = 1;
			for (int j : new Range(i + 1, domainSizes.size()))
			{
				product *= domainSizes.get(j);
			}
			
			long value = remainder / product;
			result.add((int) value);
			remainder -= value * product;
			
		}
		return result;
	}

	@Override
	public void remove()
	{
		throw new UnsupportedOperationException();		
	}

	@Override
	public Iterator<List<Integer>> iterator()
	{
		return this;
	}

}
