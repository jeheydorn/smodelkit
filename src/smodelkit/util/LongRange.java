package smodelkit.util;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class LongRange implements Iterable<Long>, Iterator<Long>
{
	long current;
	long max;
	
	public LongRange(long max)
	{
		if (max < 0)
			throw new IllegalArgumentException("Max must be at least 0.");
		current = 0;
		this.max = max;
	}
	
	/**
	 * Creates an iterator to iterate from min (inclusive) to max (exclusive).
	 */
	public LongRange(long min, long max)
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
	public Long next()
	{
		return current++;
	}

	@Override
	public void remove()
	{
		throw new UnsupportedOperationException();
	}

	@Override
	public Iterator<Long> iterator()
	{
		return this;
	}
	
	/**
	 * Iterate through all values omitted by this Range and return them in a list.
	 * @return
	 */
	public List<Long> toList()
	{
		return Helper.iteratorToList(this);
	}
	
	/**
	 * Convert this range to a stream.
	 * @return
	 */
	public Stream<Long> stream()
	{
		return StreamSupport.stream(spliterator(), false);
	}
}
