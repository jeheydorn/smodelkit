package smodelkit.util;

import java.util.Iterator;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Yields pairs of values from 2 Iterators simultaneously. The 2 iterators must be the same size.
 * @author joseph
 *
 * @param <F>
 * @param <S>
 */
public class Tuple2Iterator<F, S> implements Iterable<Tuple2<F, S>>, Iterator<Tuple2<F, S>>
{

	private Iterator<F> fIterator;
	private Iterator<S> sIterator;

	public Tuple2Iterator(Iterable<F> firstIterable, Iterable<S> secondIterable)
	{
		this.fIterator = firstIterable.iterator();
		this.sIterator = secondIterable.iterator();
	}

	@Override
	public boolean hasNext()
	{
		// I use an || so that next() will throw an exception if the 2 iterators are not the same size.
		return fIterator.hasNext() || sIterator.hasNext();
	}

	@Override
	public Tuple2<F, S> next()
	{
		return new Tuple2<F, S>(fIterator.next(), sIterator.next());
	}
	
	public Stream<Tuple2<F, S>> stream()
	{
		return StreamSupport.stream(spliterator(), false);
	}

	@Override
	public Iterator<Tuple2<F, S>> iterator()
	{
		return this;
	}

}
