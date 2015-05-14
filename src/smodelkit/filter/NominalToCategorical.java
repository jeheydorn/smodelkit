package smodelkit.filter;

import java.util.ArrayList;
import java.util.List;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Helper;

/**
 * This converts nominal attributes to binary real attributes. Continuous attributes
 * are not changed. 
 * (meaning 0.0 for the first value, and 1.0 for the second value). Nominal attributes
 * with more than 2 possible values are converted to a categorical distribution using a 
 * one of n encoding. This means that for an attribute with n nominal values, n real
 * attributes will be created, and the one corresponding to the value from the original
 * label will be set to 1.0. Others will be set to 0.0.
 * 
 * When unfiltering, the nominal value chosen is the one whose corresponding real value
 * is highest.
 *
 * If binaryToOneValue is true, then Binary nominal attributes are 
 * converted to a Bernoulli distribution rather than 2 separate real values.
 * 
 */

public class NominalToCategorical extends Filter
{
	private static final long serialVersionUID = 1L;
	private List<Integer> featureVals;
	private int totalInputVals;
	private List<Integer> labelVals;
	private int totalLabelVals;
	// These store metadata about the original inputs and labels. I need these metadata
	// matrixes so that I can tell which attributes are continuous vs binary.
	private Matrix inputsMeta;
	private Matrix labelsMeta;
	/**
	 * Convert binary values 1 real value instead of 2.
	 */
	private boolean binaryToOneValue;
	
	public NominalToCategorical()
	{
		this.binaryToOneValue = false;
		totalInputVals = 0;
		labelVals = null;
		totalLabelVals = 0;
	}

	public NominalToCategorical(boolean binaryToOneValue)
	{
		this.binaryToOneValue = binaryToOneValue;
		totalInputVals = 0;
		labelVals = null;
		totalLabelVals = 0;
	}

	// Decide how many dims are needed for each column
	public void initializeInternal(Matrix inputs, Matrix labels)
	{
		inputsMeta = new Matrix();
		inputsMeta.copyMetadata(inputs);
		labelsMeta = new Matrix();
		labelsMeta.copyMetadata(labels);
		
		// Count the new feature dims
		featureVals = new ArrayList<Integer>(inputs.cols());
		labelVals = new ArrayList<Integer>(labels.cols());
		totalInputVals = 0;
		totalLabelVals = 0;
		for(int i = 0; i < inputs.cols(); i++)
		{
			int n = inputs.getValueCount(i);
			if(n == 0 || (binaryToOneValue && n == 2))
			{
				n = 1;
			}
			
			featureVals.add(n);
			totalInputVals += n;
		}
		
		// Count the new label dims
		for(int i = 0; i < labels.cols(); i++)
		{
			int n = labels.getValueCount(i);
			if(n == 0 || (binaryToOneValue && n == 2))
			{
				n = 1;
			}
			
			labelVals.add(n);
			totalLabelVals += n;
		}	
	}

	// Convert categorical distributions back to nominal values (by finding the mode)
	protected Vector unfilterLabelInternal(Vector before)
	{
		if(before.size() != totalLabelVals)
			throw new IllegalArgumentException("Unexpected label vector size");
		
		double[] after = new double[labelVals.size()];
		int curLabelStart = 0;
		
		for (int i = 0; i < after.length; i++)
		{
			if(labelsMeta.isContinuous(i)) // If the label is continuous...
			{
				// Continuous labels are just copied straight across
				after[i] = before.get(curLabelStart);
				curLabelStart++;
			}
			else if (labelsMeta.getValueCount(i) == 2 && binaryToOneValue)
			{
				// Binary nominal attribute stored as one value.
				after[i] = Math.round(before.get(curLabelStart));
				curLabelStart++;
			}
			else
			{
				// Nominal attribute more than binary.
				
				// Find the mode, and use it for the predicted nominal label
				double maxIndex = Helper.indexOfMaxElementInRange(before, curLabelStart, labelVals.get(i));
				
				after[i] = (double)(maxIndex - curLabelStart);
				curLabelStart += labelVals.get(i);
			}
		}
		
		return new Vector(after, before.getWeight());
	}


	// Convert nominal features in the training set to categorical distributions
	protected Matrix filterInputsInternal(Matrix inputs)
	{
		Matrix result = new Matrix();
		result.setSize(0, totalInputVals);
		for(int i = 0; i < inputs.rows(); i++)
		{
			Vector row = filterInputInternal(inputs.row(i));
			result.addRow(row);
		}
		
		result.setNumCatagoricalCols(featureVals);
		return result;
	}


	// Convert each label to a categorical distribution
	protected Matrix filterLabelsInternal(Matrix labels)
	{	
		Matrix result = new Matrix();
		result.setSize(0, totalLabelVals);
		for(int i = 0; i < labels.rows(); i++)
		{
			Vector row = filterLabelInternal(labels.row(i));
			result.addRow(row);
		}
		
		result.setNumCatagoricalCols(labelVals);
	
		return result;
	}

	/**
	 *  Convert all nominal values to a categorical distribution
	 */
	@Override
	protected Vector filterInputInternal(Vector before)
	{
		return filterRow(before, inputsMeta);
	}
	
	@Override
	public Vector filterLabelInternal(Vector before)
	{
		return filterRow(before, labelsMeta);
	}
	
	private Vector filterRow(Vector before, Matrix matrixMeta)
	{
		if (before.size() != matrixMeta.cols())
			throw new IllegalArgumentException("Unexpected vector size");
		List<Double> after = new ArrayList<Double>();
		for (int i = 0; i < before.size(); i++)
		{
			if(matrixMeta.isContinuous(i)) // If the value is continuous...
			{
				// Continuous values are just copied straight across
				after.add(before.get(i));
			}
			else
			{
				// Nominal values are converted to a categorical distribution, unless they are binary, in which
				// case they are made into a Bernoulli distribution.
				if (matrixMeta.getValueCount(i) == 2 && binaryToOneValue)
				{
					// Create a Bernoulli distribution.
					if(Vector.isUnknown(before.get(i)))
					{
						after.add(Vector.getUnknownValue());
					}
					else
					{
						// Binary nominal values are internally stored as 0 and 1, so I don't need to change
						// the value.
						after.add(before.get(i));
					}
				}
				else
				{
					// Create a categorical distribution.

					if(Vector.isUnknown(before.get(i)))
					{					
						// One missing value becomes multiple missing values.
						for(int j = 0; j < matrixMeta.getValueCount(i); j++)
							after.add(Vector.getUnknownValue());
					}
					else
					{
						// Give all probability to the one value.
						int v = (int)before.get(i);
						if (v < 0)
							throw new IllegalArgumentException("Nominal attribute values cannot be negative.");
						if(v >= matrixMeta.getValueCount(i))
							throw new IllegalArgumentException("Attribute value is greater than the number of nominal"
									+ " values. Value: " + v + ", # nominal values: " + matrixMeta.getValueCount(i));
						int pos = after.size();
						for(int j = 0; j < matrixMeta.getValueCount(i); j++)
						{
							after.add(0.0);						
						}
						after.set(pos + v, 1.0);
					}
				}
			}
		}
		return new Vector(Helper.toDoubleArray(after), before.getWeight());
	}

	@Override
	public void configure(String[] args)
	{
		for (String arg : args)
		{
			if (arg.equals("--binaryToOneValue") || arg.equals("-b"))
			{
				binaryToOneValue = true;
			}
		}
		
	}
	
}
