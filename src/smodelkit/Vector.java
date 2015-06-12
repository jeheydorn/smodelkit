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
public interface Vector extends Serializable, Comparable<Vector>
{
	/**
	 * Creates a new vector with instance weight 1.
	 * @param values The values to be stored within the vector.
	 */
	public static Vector create(double... values)
	{
		if (Matrix.useDouble)
			return new VectorDouble(values);
		else
			return new VectorFloat(convertToFloats(values));
	}

	/**
	 * Creates a new vector with instance weight 1.
	 * @param values The values to be stored within the vector.
	 */
	public static Vector create(float... values)
	{
		if (Matrix.useDouble)
			return new VectorDouble(convertToDoubles(values));
		else
			return new VectorFloat(values);
	}
	
	public static Vector create(double[] values, double weight)
	{
		if (Matrix.useDouble)
			return new VectorDouble(values, weight);
		else
			return new VectorFloat(convertToFloats(values), (float)weight);		
		
	}

	public static Vector create(float[] values, float weight)
	{
		if (Matrix.useDouble)
			return new VectorDouble(convertToDoubles(values), (float)weight);
		else
			return new VectorFloat(values, weight);	
		
	}

	public static Vector create(double[] values, double weight, int from, int to)
	{
		if (Matrix.useDouble)
			return new VectorDouble(values, weight, from, to);
		else
			return new VectorFloat(convertToFloats(values), (float)weight, from, to);		
	}

	public static Vector create(float[] values, float weight, int from, int to)
	{
		if (Matrix.useDouble)
			return new VectorDouble(convertToDoubles(values), weight, from, to);
		else
			return new VectorFloat(values, weight, from, to);		
	}
	
	/**
	 * Creates a copy of other.
	 */
	public static Vector create(Vector other)
	{
		if (Matrix.useDouble)
			return new VectorDouble((VectorDouble) other);
		else
			return new VectorFloat((VectorFloat)other);
	}
	
	/**
	 * Creates a new Vector with values from other, and the specified weight.
	 */
	public static Vector create(Vector other, double weight)
	{
		if (Matrix.useDouble)
			return new VectorDouble((VectorDouble)other, weight);
		else
			return new VectorFloat((VectorFloat)other, (float)weight);					
	}

	/**
	 * Creates a new Vector with values from other, and the specified weight.
	 */
	public static Vector create(Vector other, float weight)
	{
		if (Matrix.useDouble)
			return new VectorDouble((VectorDouble)other, weight);
		else
			return new VectorFloat((VectorFloat)other, weight);					
	}

	public double getWeight();
	
	public float getWeightFloat();
	
	public void setWeight(double value);
	
	public void setWeight(float value);
		
	public double get(int index);	
	
	public float getFloat(int index);
	
	/**
	 * Returns the internal values from this vector.
	 * The caller MUST NOT modify these values.
	 */
	public double[] getValues();
	
	public float[] getValuesFloat();

	public void set(int index, double value);
	
	public void set(int index, float value);
		
	/**
	 * Creates a new vector which gives a view of the values from this vector in the specified range.
	 * This runs in O(1) time with respect to the size of this vector.
	 * @param from (inclusive).
	 * @param to (exclusive).
	 * @return
	 */
	public Vector subVector(int from, int to);
	
	public int size();
	
	/**
	 * Concatenates the values from the given vector the the end of this vector.
	 * @param v
	 */
	public void addAll(Vector v);
	
	/**
	 * Returns a new vector with the values of this vector concatenated with the values of v.
	 * The weight of the result is that this.
	 */
	public Vector concat(Vector v);

	/**
	 * Returns a new vector with the values of this vector concatenated with the values of v.
	 * The weight of the result is that this.
	 */
	public Vector concat(double[] v);
	
	public Vector concat(float[] v);

	/**
	 * Removes the value at the specified index.
	 */
	public void remove(int index);
			
	/**
	 * Determines if the given value represents an unknown (missing) value.
	 */
	public static boolean isUnknown(double value)
	{
		return Double.isNaN(value);
	}

	/**
	 * Determines if the given value represents an unknown (missing) value.
	 */
	public static boolean isUnknown(float value)
	{
		return Float.isNaN(value);
	}
	
	public static double getUnknownValue()
	{
		return Double.NaN;
	}

	public static double getUnknownValueFloat()
	{
		return Float.NaN;
	}

	/**
	 * Compares vector values. Ignores weight.
	 */
	@Override
	public boolean equals(Object other);
			
	int getFrom();
	
	int getTo();
	
	public static double[] convertToDoubles(float[] array)
	{
		double[] result = new double[array.length];
		for (int i = 0; i < array.length; i++)
		{
			result[i] = array[i];
		}
		return result;
	}

	public static float[] convertToFloats(double[] array)
	{
		float[] result = new float[array.length];
		for (int i = 0; i < array.length; i++)
		{
			result[i] =(float) array[i];
		}
		return result;
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
		assertEquals(v1.getWeight(), v2.getWeight(), threshold);
	}

}
