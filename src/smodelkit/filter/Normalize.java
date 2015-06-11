package smodelkit.filter;

import java.util.Set;
import java.util.TreeSet;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;

public class Normalize extends Filter
{
	private static final long serialVersionUID = 1L;
	private double[] featureMins;
	private double[] featureMaxes;
	private double[] labelMins;
	private double[] labelMaxes;
	private Set<Integer> ignoredInputAttributes;
	// base is the lowest value after filtering.
	private static final double base = -1.0;
	// range is the largest minus the smallest value after filtering.
	private static final double range = 2.0;
	
	public Normalize()
	{
		ignoredInputAttributes = new TreeSet<>();
	}
	
	public void ignoreInputAttribute(int column)
	{
		if (column < 0)
			throw new IllegalArgumentException("Invalid column to ignore: " + column);
		ignoredInputAttributes.add(column);
	}
	
	// Computes the min and max of each column
	@Override
	public void initializeInternal(Matrix inputs, Matrix labels)
	{		
		featureMins = new double[inputs.cols()];
		featureMaxes = new double[inputs.cols()];
		for(int i = 0; i < inputs.cols(); i++)
		{
			if(inputs.getValueCount(i) == 0 && !ignoredInputAttributes.contains(i))
			{
				// Compute the min and max
				featureMins[i] = inputs.findMin(i);
				featureMaxes[i] = inputs.findMax(i);
			}
			else
			{
				// Don't do nominal attributes and ignored columns.
				featureMins[i] = Vector.getUnknownValue();
				featureMaxes[i] = Vector.getUnknownValue();
			}
		}
		
		labelMins = new double[labels.cols()];
		labelMaxes = new double[labels.cols()];
		for(int i = 0; i < labels.cols(); i++)
		{
			if(labels.getValueCount(i) == 0)
			{
				// Compute the min and max
				labelMins[i] = labels.findMin(i);
				labelMaxes[i] = labels.findMax(i);
			}
			else
			{
				// Don't do nominal attributes
				labelMins[i] = Vector.getUnknownValue();
				labelMaxes[i] = Vector.getUnknownValue();
			}
		}

	}

	/**
	 *  Normalize continuous values to fall from base to (base + range).
	 */
	protected Vector filterInputInternal(Vector before)
	{
		return filterInternal(before, featureMins, featureMaxes);
	}
	
	public Vector filterLabelInternal(Vector before)
	{
		return filterInternal(before, labelMins, labelMaxes);
	}

	/**
	 * For filtering either inputs or labels. The range of the filtered values will be between
	 * base and (base + range) inclusive.
	 */
	private static Vector filterInternal(Vector before, double[] mins, double[] maxes)
	{
		if (before == null)
			throw new IllegalArgumentException();
		if (mins == null)
			throw new IllegalArgumentException();
		if(before.size() != mins.length)
			throw new IllegalArgumentException(String.format("Unexpected input vector size. " +
					"Expected size %d but was size %d", mins.length, before.size()));
		double[] after = new double[before.size()];
		for(int c = 0; c < mins.length; c++)
		{
			if(Vector.isUnknown(mins[c])) // if the attribute is nominal or ignored...
				after[c] = before.get(c);
			else
			{
				if(Vector.isUnknown(before.get(c)))
				{
					after[c] = Vector.getUnknownValue();
				}
				else
				{
					if (mins[c] != maxes[c])
					{
						after[c] = ((before.get(c) - mins[c]) / (maxes[c] - mins[c]) * range) + base;					
					}
					else
					{
						// There is only 1 value in this column.
						if (before.get(c) > 0)
							after[c] = base + range;
						else
							after[c] = base;
					}
				}
			}
		}
		return new VectorDouble(after, before.getWeight());
	}


	/**
	 * De-normalize continuous values back to their original range
	 */
	protected Vector unfilterLabelInternal(Vector before)
	{
		double[] after = new double[labelMaxes.length];

		for (int c = 0; c < after.length; c++)
		{
			if(Vector.isUnknown(labelMins[c])) // if the attribute is nominal or ignored...
				after[c] = before.get(c);
			else
			{
				if (labelMaxes[c] == labelMins[c])
				{
					// There was only 1 value in this column when the filter was initialized.
					after[c] = labelMaxes[c];
				}
				else
				{
					after[c] = ((before.get(c) - base) / range) * (labelMaxes[c] - labelMins[c]) + labelMins[c];
				}
			}
		}
		return new VectorDouble(after, before.getWeight());
	}
	@Override
	public void configure(String[] args)
	{
	}
};
