package smodelkit.util;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class Range implements Iterable<Integer>, Iterator<Integer>
{
	int current;
	int max;
	
	public Range(int max)
	{
		if (max < 0)
			throw new IllegalArgumentException("Max must be at least 0.");
		current = 0;
		this.max = max;
	}
	
	/**
	 * Creates an iterator to iterate from min (inclusive) to max (exclusive).
	 */
	public Range(int min, int max)
	{
		if (min > max)
			throw new IllegalArgumentException("Min must be less than or equal to max.");
		current = min;
		this.max = max;
	}
	
	@Override
	public boolean hasNext()
	{
		return current < max;
	}

	@Override
	public Integer next()
	{
		return current++;
	}

	@Override
	public void remove()
	{
		throw new UnsupportedOperationException();
	}

	@Override
	public Iterator<Integer> iterator()
	{
		return this;
	}
	
	/**
	 * Iterate through all values omitted by this Range and return them in a list.
	 * @return
	 */
	public List<Integer> toList()
	{
		return Helper.iteratorToList(this);
	}
	
	/**
	 * Convert this range to a stream.
	 * @return
	 */
	public Stream<Integer> stream()
	{
		return StreamSupport.stream(spliterator(), false);
	}
}
