package smodelkit.evaluator;

import java.util.ArrayList;
import java.util.List;

import smodelkit.ConfusionMatrix;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Helper;
import smodelkit.util.Range;

/**
 * Gives accuracy per column. Like hamming accuracy except not averaged over all columns.
 * @author joseph
 *
 */
public class AccuracyPerColumn extends Evaluator
{	
	private static final long serialVersionUID = 1L;
	private double[] correctCounts;
	private int totalCount;
	private List<ConfusionMatrix> confusions;
	Matrix metadata;
	
	public AccuracyPerColumn()
	{
	}
	
	@Override
	public void configure(String[] args)
	{		
	}
		
	@Override
	protected void startBatch(Matrix metadata)
	{
		// Make sure we don't have continuous labels.
		if (metadata.cols() > 1)
		{
			for (int i = 0; i < metadata.cols(); i++)
			{
				if (metadata.isContinuous(i))
					throw new IllegalArgumentException("HammingAccuracy does not work on continous labels.");
			}
		}
		
		correctCounts = new double[metadata.cols()];
		totalCount = 0;
		
		confusions = new ArrayList<>();
		for (int c = 0; c < metadata.cols(); c++)
		{
			confusions.add(new ConfusionMatrix(metadata.getAttrName(c)));
		}

		
		this.metadata = new Matrix();
		this.metadata.copyMetadata(metadata);
	}

	@Override
	protected void evaluate(Vector target, List<Vector> predictions)
	{
		Vector prediction = predictions.get(0);
		assert target.size() == prediction.size();

		// Update confusions.
		for (int lCol = 0; lCol < target.size(); lCol++)
		{
			// increment the count corresponding to the prediction.
			int targValue = (int) (Vector.isUnknown(target.get(lCol)) ? Double.MAX_VALUE : target.get(lCol));
			int predValue = (int) (Vector.isUnknown(prediction.get(lCol)) ? Double.MAX_VALUE : prediction.get(lCol));
			confusions.get(lCol).incrementCount(
					metadata.getAttrValueName(lCol, targValue), metadata.getAttrValueName(lCol, predValue));
		}

		for (int lCol : new Range(target.size()))
		{
			if (target.get(lCol) == prediction.get(lCol))
				correctCounts[lCol]++;
		}
		
		totalCount++;
	}

	@Override
	protected List<Double> calcScores()
	{
		double[] dividedCounts = new double[correctCounts.length];
		for (int lCol : new Range(correctCounts.length))
		{
			dividedCounts[lCol] = correctCounts[lCol] / totalCount;
		}
		
		return Helper.toDoubleList(dividedCounts);
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
	
	@Override
	public boolean higherScoresAreBetter()
	{
		return true;
	}



};
