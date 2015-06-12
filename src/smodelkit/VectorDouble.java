package smodelkit;

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
public class VectorDouble implements Serializable, Comparable<Vector>, Vector
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
	protected VectorDouble(double... values)
	{
		this.values = values;
		this.weight = 1.0;
		to = values.length;
		varify();
	}
		
	protected VectorDouble(double[] values, double weight)
	{
		this.values = values;
		this.weight = weight;
		to = values.length;
		varify();
	}
	
	protected VectorDouble(double[] values, double weight, int from, int to)
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

	protected VectorDouble(VectorDouble other)
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
	protected VectorDouble(VectorDouble other, double weight)
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
	 * Returns the values in this vector as an array.
	 * The caller MUST NOT modify these values.
	 */
	public double[] getValues()
	{
		compact();
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
	public VectorDouble subVector(int from, int to)
	{
		return new VectorDouble(values, weight, this.from + from, this.from + to);
		
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
		VectorDouble result = new VectorDouble(this);
		result.addAll(v);
		return result;
	}

	/**
	 * Returns a new vector with the values of this vector concatenated with the values of v.
	 * The weight of the result is that this.
	 */
	public VectorDouble concat(double[] v)
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
		return new VectorDouble(result, weight);
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
	 * Compares vector values. Ignores weight.
	 */
	@Override
	public boolean equals(Object other)
	{
		if (!(other instanceof VectorDouble))
			return false;
		VectorDouble otherV = (VectorDouble)other;
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
		
	@Override
	public String toString()
	{
		if (weight == 1.0)
		{
			return valuesToString();
		}
		else
		{
			return "VectorDouble: values: " + valuesToString() + ", weight: " + weight;
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
	
	@Override
	public float getWeightFloat()
	{
		return (float)weight;
	}
	
	@Override
	public void setWeight(float value)
	{
		weight = value;
	}
	
	@Override
	public float getFloat(int index)
	{
		return (float)values[index];
	}
	
	@Override
	public float[] getValuesFloat()
	{
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void set(int index, float value)
	{
		set(index, (double)value);
	}
	
	@Override
	public Vector concat(float[] v)
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
		return new VectorDouble(result, weight);
	}
	
	@Override
	public int getFrom()
	{
		return from;
	}
	@Override
	public int getTo()
	{
		return to;
	}
	
}
