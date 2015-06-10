package smodelkit;

import static org.junit.Assert.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.DoubleStream;

import smodelkit.util.Range;



/**
 * A vector of double values. This can be used as an instance, or part of an instance.
 * 
 * @author joseph
 *
 */
@SuppressWarnings("serial")
public class Vector implements Serializable, Comparable<Vector>
{
	/**
	 * Elements of this array should never be modified because they may be shared by other Vectors.
	 */
	private double[] values;
	

	private int from;

	private int to;
	
	/**
	 * This is the instance weight when training, and a predicted confidence level when predicting
	 * scored lists.
	 */
	private double weight;
	public double getWeight()
	{
		return weight;
	}
	public void setWeight(double value)
	{
		if (value < 0.0)
			throw new IllegalArgumentException("Instance weights cannot be negative. value: " + value);
		weight = value;
	}
	
	/**
	 * Creates a new vector with instance weight 1.
	 * @param values The values to be stored within the vector.
	 */
	public Vector(double... values)
	{
		this.values = values;
		this.weight = 1.0;
		to = values.length;
		varify();
	}
		
	public Vector(double[] values, double weight)
	{
		this.values = values;
		this.weight = weight;
		to = values.length;
		varify();
	}
	
	private Vector(double[] values, double weight, int from, int to)
	{
		if (from >= to)
			throw new IllegalArgumentException("Bad range.");
		if (to > values.length)
			throw new IllegalArgumentException("Bad range.");
		if (from < 0)
			throw new IllegalArgumentException("Bad range.");
			
		this .values = values;
		this.weight = weight;
		this.from = from;
		this.to = to;
	}

	public Vector(Vector other)
	{
		this.values = other.values;
		this.weight = other.weight;
		this.from = other.from;
		this.to = other.to;
		varify();
	}
	
	/**
	 * Creates a new Vector with values from other, and the specified weight.
	 */
	public Vector(Vector other, double weight)
	{
		this.values = other.values;
		this.weight = weight;
		to = values.length;
		varify();
	}
	
	private void varify()
	{
		if (weight < 0.0)
			throw new IllegalArgumentException("Instance weights cannot be negative.");
	}

	public double get(int index)
	{
		return values[from + index];
	}
	
	/**
	 * Returns the internal values from this vector.
	 * The caller MUST NOT modify these values.
	 */
	public double[] getValues()
	{
		if (!isCompact())
			throw new IllegalStateException();
		return values;
	}
	
	/**
	 * Sets the value at the specified index to the specified value.
	 * 
	 * The internal array is copied to avoid changing it.
	 */
	public void set(int index, double value)
	{
		if (!compact())
		{
			values = Arrays.copyOf(values, values.length);
		}
		values[index] = value;
	}
	
	private boolean isCompact()
	{
		return from == 0 && to == values.length;
	}
	
	private boolean compact()
	{
		if (!isCompact())
		{
			values = Arrays.copyOfRange(values, from, to);
			from = 0;
			to = values.length;
			return true;
		}
		return false;
	}
	
	/**
	 * Creates a new vector which gives a view of the values from this vector in the specified range.
	 * This runs in O(1) time with respect to the size of this vector.
	 * @param from (inclusive).
	 * @param to (exclusive).
	 * @return
	 */
	public Vector subVector(int from, int to)
	{
		return new Vector(values, weight, this.from + from, this.from + to);
		
	}
	
	public int size()
	{
		return to - from;
	}
	
	/**
	 * Concatenates the values from the given vector the the end of this vector.
	 * @param v
	 */
	public void addAll(Vector v)
	{
		double[] newValues = new double[size() + v.size()];
		for (int i = 0; i < size(); i++)
		{
			newValues[i] = get(i);
		}
		for (int i = 0; i < v.size(); i++)
		{
			newValues[i + size()] = v.get(i);
		}
		this.values = newValues;
		from = 0;
		to = values.length;
	}
	
	/**
	 * Returns a new vector with the values of this vector concatenated with the values of v.
	 * The weight of the result is that this.
	 */
	public Vector concat(Vector v)
	{
		Vector result = new Vector(this);
		result.addAll(v);
		return result;
	}

	/**
	 * Returns a new vector with the values of this vector concatenated with the values of v.
	 * The weight of the result is that this.
	 */
	public Vector concat(double[] v)
	{
		double[] result = new double[size() + v.length];
		for (int i = 0; i < size(); i++)
		{
			result[i] = get(i);
		}
		for (int i = 0; i < v.length; i++)
		{
			result[i + size()] = v[i];
		}
		return new Vector(result, weight);
	}

	/**
	 * Removes the value at the specified index.
	 */
	public void remove(int index)
	{
		double[] temp = new double[size() - 1];
		for (int i = 0; i < temp.length; i++)
		{
			if (i < index)
				temp[i] = get(i);
			else
				temp[i] = get(i + 1);
		}
		values = temp;
		from = 0;
		to = values.length;
	}
		
	/**
	 * Returns the value which represents unknown values. If you need to know if a value
	 * is unknown, use isUnknown(value). Do NOT use value == getUnknownValue().
	 * @return
	 */
	public static double getUnknownValue()
	{
		return Double.NaN;
	}
	
	/**
	 * Determines if the given value represents an unknown (missing) value.
	 */
	public static boolean isUnknown(double value)
	{
		return Double.isNaN(value);
	}
		
	/**
	 * Compares vector values. Ignores weight.
	 */
	@Override
	public boolean equals(Object other)
	{
		if (!(other instanceof Vector))
			return false;
		Vector otherV = (Vector)other;
		if (size() != otherV.size())
			return false;
		for (int i = 0; i < size(); i++)
		{
			if (get(i) != otherV.get(i))
				return false;
		}
		return true;
	}
	
	@Override
	/**
	 * Compares vector values. Ignores weight.
	 */
	public int compareTo(Vector v)
	{
		for (int i = 0; i < size() && i < v.size(); i++)
		{				
			if (get(i) < v.get(i))
				return -1;
			if (get(i) > v.get(i))
				return 1;
		}
		
		int sizeComp = Integer.compare(size(), v.size());
		return sizeComp;
		
	}
	
	public DoubleStream stream()
	{
		return Arrays.stream(values, from, to);
	}
	
	@Override
	public String toString()
	{
		if (weight == 1.0)
		{
			return valuesToString();
		}
		else
		{
			return "values: " + valuesToString() + ", weight: " + weight;
		}
	}
	
	private String valuesToString()
	{
		StringBuilder result = new StringBuilder();
		result.append("[");
		for (int i : new Range(size()))
		{
			result.append(get(i));
			if (i < size() - 1)
				result.append(", ");
		}
		result.append("]");
		return result.toString();
	}
	
	public static void assertVectorEquals(Vector v1,  Vector v2, double threshold)
	{
		if (v1.size() != v2.size())
		{
			throw new AssertionError("Vectors differ in size. v1 size=" + v1.size() + ", v2 size=" + v2.size());
		}
		for (int i : new Range(v1.size()))
		{
			if (Math.abs(v1.get(i) - v2.get(i)) > threshold)
			{
				throw new AssertionError("Vectors differ. v1.get(" + i + ")=" + v1.get(i) + ", v2.get(" + i + ")=" + v2.get(i));
			}
			assertEquals(v1.get(i), v2.get(i), threshold);
		}
		assertEquals(v1.weight, v2.weight, threshold);
	}
	
}
