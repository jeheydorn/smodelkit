package smodelkit.filter;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Range;

/**
 * Replaces unknown input features values with their means/modes for that attribute.
 * @author joseph
 *
 */
public class MeanModeUnknownFiller extends Filter
{
	private static final long serialVersionUID = 1L;
	
	double[] meanModes;
	
	public MeanModeUnknownFiller()
	{
	}
	
	@Override
	public void initializeInternal(Matrix inputs, Matrix trainLabels)
	{
		// Find the mean/mode of each column.
		meanModes = new double[inputs.cols()];
		
		if (inputs.rows() == 0)
		{
			System.err.println("Warning: MeanModeUnknownFiller was given an empty dataset to initialize with.");
			return;
		}
		
		for (int c : new Range(meanModes.length))
		{
			if (inputs.isContinuous(c))
				meanModes[c] = inputs.findMean(c);
			else
				meanModes[c] = inputs.findMode(c);
		}
	}
	
	@Override
	protected Vector filterInputInternal(Vector before)
	{
		if (before.size() != meanModes.length)
			throw new IllegalArgumentException("Expected input length " + meanModes.length
					+ ", but was " + before.size());
		
		// If before has no unknown values, return it to save memory.
		if (!before.stream().anyMatch(d -> Vector.isUnknown(d)))
			return before;
		
		double[] after = new double[before.size()];
		for (int i : new Range(before.size()))
		{
			if (Vector.isUnknown(before.get(i)))
				after[i] = meanModes[i];
			else
				after[i] = before.get(i);
		}
		return new Vector(after, before.getWeight());
	}

	@Override
	protected Vector unfilterLabelInternal(Vector before)
	{
		return before;
	}

	@Override
	protected Matrix filterLabelsInternal(Matrix labels)
	{
		return labels;
	}

	@Override
	protected Vector filterLabelInternal(Vector before)
	{
		return before;
	}

	@Override
	public void configure(String[] args)
	{		
	}

}
