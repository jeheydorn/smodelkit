package smodelkit.util;

public class BitSet
{
	long lower;
	long upper;
	
	public BitSet()
	{
		// Empty set.
	}
	
	public void add(byte value)
	{
		if (value >= 0 && value < 64)
		{
			lower |= 1L << value;
		}
		else if (value >= 64)
		{
			upper |= 1L << (value - 64);
		}
		else
		{
			throw new IllegalArgumentException("Value out of range.");
		}
	}
	
	public void remove(byte value)
	{
		if (value >= 0 && value < 64)
		{
			lower &= ~(1L << value);
		}
		else if (value >= 64)
		{
			upper &= ~(1L << (value - 64));
		}
		else
		{
			throw new IllegalArgumentException("Value out of range.");
		}		
	}

	public boolean contains(byte value)
	{
		if (value >= 0 && value < 64)
		{
			return (lower & (1L << value)) != 0;
		}
		else if (value >= 64)
		{
			return (upper & (1L << (value - 64))) != 0;
		}
		else
		{
			throw new IllegalArgumentException("Value out of range.");
		}		
	}
	
	public static void main(String[] args)
	{
		BitSet set = new BitSet();
		set.add((byte) 1);
		set.add((byte) 2);
		set.add((byte) 120);
		
		System.out.println(set.contains((byte) 2));
		System.out.println(set.contains((byte) 3));
		System.out.println(set.contains((byte) 120));
		System.out.println(set.contains((byte) 121));
	}

}
