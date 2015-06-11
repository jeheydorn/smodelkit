package smodelkit.evaluator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

import smodelkit.ConfusionMatrix;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.util.Range;

/**
 * Finds exact match accuracy of a specified group of outputs.
 * @author joseph
 *
 */
public class AccuracyOfGroup extends Evaluator
{	
	private static final long serialVersionUID = 1L;
	List<String> relevantColumns;
	private List<ConfusionMatrix> confusions;
	private int correctCount;
	private int totalCount;
	private List<Integer> relevantColumnIndexes;
	Matrix metadata;

	public AccuracyOfGroup(List<String> relevantColumns) 
	{
		this.relevantColumns = relevantColumns;
		
		// Check for duplicate attribute names.
		Set<String> prev = new TreeSet<>();
		for (String name : this.relevantColumns)
		{
			if (prev.contains(name))
				throw new IllegalArgumentException("Dupilcate attribute name: " + name);
			prev.add(name);
		}
	}
	
	public AccuracyOfGroup()
	{
	}
	
	@Override
	public void configure(String[] args)
	{
		relevantColumns = Arrays.asList(args);
	
	}
	
	private static Vector selectRelevantColumns(List<Integer> relevantColumnIndexes, Vector label)
	{
		double[] result = new double[relevantColumnIndexes.size()];
		int i = 0;
		for (int c : relevantColumnIndexes)
		{
			result[i] = label.get(c);
			i++;
		}
		return Vector.create(result);
	}

	@Override
	public boolean higherScoresAreBetter()
	{
		return true;
	}

	@Override
	protected void startBatch(Matrix metadata)
	{
		// Make sure all relevant outputs are nominal.
		for (String attrName : relevantColumns)
		{
			int c = metadata.getAttributeColumnIndex(attrName);
			if (c == -1)
				throw new IllegalArgumentException("Unknown attribute name: " + attrName);
			if (metadata.isContinuous(c))
				throw new IllegalArgumentException("AccuracyOfGroup does not work on continuous outputs.");
		}
		
		confusions = new ArrayList<>();
		for (String attrName : relevantColumns)
		{
			confusions.add(new ConfusionMatrix(attrName));
		}				

		correctCount = 0;
		totalCount = 0;

		relevantColumnIndexes = relevantColumns.stream().map(
				name -> metadata.getAttributeColumnIndex(name)).collect(Collectors.toList());

		this.metadata = new Matrix();
		this.metadata.copyMetadata(metadata);
	}

	@Override
	protected void evaluate(Vector target, List<Vector> predictions)
	{
		Vector prediction = selectRelevantColumns(relevantColumnIndexes, predictions.get(0));

		// Make sure the predictions are all valid nominal values.
		for (int c : new Range(prediction.size()))
		{
			if(prediction.get(c) >= metadata.getValueCount(relevantColumnIndexes.get(c)))
				throw new IllegalArgumentException("The prediction is out of range");
		}

		// Set up target values
		Vector targ = selectRelevantColumns(relevantColumnIndexes, target);

		assert targ.size() == prediction.size();

		for (int c : new Range(relevantColumnIndexes.size()))
		{
			if(confusions != null)
			{
				int lCol = relevantColumnIndexes.get(c);
				// increment the count corresponding to the prediction.
				int targValue = (int) (Vector.isUnknown(targ.get(c)) ? Double.MAX_VALUE : targ.get(c));
				int predValue = (int) (Vector.isUnknown(prediction.get(c)) ? Double.MAX_VALUE : prediction.get(c));
				confusions.get(c).incrementCount(
						metadata.getAttrValueName(lCol, targValue), metadata.getAttrValueName(lCol, predValue));
			}
		}

		if (targ.equals(prediction))
			correctCount++;
		totalCount++;
	}

	@Override
	protected List<Double> calcScores()
	{
		return Arrays.asList(((double)correctCount) / totalCount);
	}

	@Override
	protected List<ConfusionMatrix> calcConfusions()
	{
		return confusions;
	}

	@Override
	protected int getMaxDesiredSize()
	{
		return 1;
	}

	
};
