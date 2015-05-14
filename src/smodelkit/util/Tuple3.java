package smodelkit.util;

import java.io.Serializable;

/**
 * A 3-tuple of objects which don't have to be comparable.
 * @author joseph
 *
 */
public class Tuple3<F, S, T> implements Serializable
{
	private static final long serialVersionUID = 1L;
	private F first;
	private S second;
	private T third;
	
	public Tuple3(F first, S second, T third)
	{
		this.first = first;
		this.second = second;
		this.third = third;
	}
	
	public F getFirst()
	{
		return first;
	}
	
	public S getSecond()
	{
		return second;
	}
	
	public T getThird()
	{
		return third;
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (!(other instanceof Tuple3))
			return false;
		
		@SuppressWarnings("unchecked")
		Tuple3<F, S, T> otherTuple3 = (Tuple3<F, S, T>)other;
		return first.equals(otherTuple3.first) && second.equals(otherTuple3.second) && third.equals(otherTuple3.third);
	}
	
}
