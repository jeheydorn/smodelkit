package smodelkit.util;

public class Triple<F extends Comparable<F>, S extends Comparable<S>, T extends Comparable<T>>
{
	private F first;
	private S second;
	private T third;
	
	public Triple(F first, S second, T third)
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
}
