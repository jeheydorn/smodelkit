package smodelkit.evaluator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import smodelkit.ConfusionMatrix;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Range;

/**
 * Measure accuracy of learners that give multiple predictions. The model predicts
 * a ranked list of predictions. This ranked list of predictions is counted correct
 * if any of the top n output vectors in it match the target output vector exactly.
 * This is only for nominal outputs.
 * 
 * When n=1, this is exact match accuracy.
 * 
 * Confusion matrices will be created while evaluating iff only 1 value of n is used.
 * In this case, if n is 1, then the confusion matrix will be as expected for exact
 * match accuracy. If n > 1, then the confusion matrix will be updated for all n
 *  (or up to n) predictions.
 *
 */
public class TopN extends Evaluator
{
	private static final long serialVersionUID = 1L;
	private List<Integer> nValues;
	private ArrayList<ConfusionMatrix> confusions;
	
	// Maps n values to numbers of correct predictions.
	private Map<Integer, Integer> correctCounts;
	private int totalCount;
	Matrix metadata;
	
	private int largestN;
	
	public TopN()
	{
	}

	public TopN(List<Integer> ns)
	{
		configure(ns);
	}

	/**
	 * @param ns The n in n-best. The first entry is used to compute the accuracy that will be returned.
	 * Subsequent entries will be calculated and printed out, but not returned. 
	 */
	public void configure(List<Integer> ns)
	{
		this.nValues = ns;
		
		if (ns.isEmpty())
			throw new IllegalArgumentException("No values of n specified.");
		for (Integer n : ns)
		{
			if (n == null || n < 1)
				throw new IllegalArgumentException("Bad n value: " + n);
		}
		
		largestN = Collections.max(ns);
	}
	
	@Override
	public void configure(String[] args)
	{
		List<Integer> ns = new ArrayList<Integer>();
		for (String arg : args)
		{
			ns.add(Integer.parseInt(arg));
		}
		configure(ns);
	}
	
	@Override
	public boolean higherScoresAreBetter()
	{
		return true;
	}

	@Override
	protected void startBatch(Matrix metadata) 
	{
		// Make sure we don't have continuous outputs.
		for (int i = 0; i < metadata.cols(); i++)
		{
			if (metadata.isContinuous(i))
				throw new UnsupportedOperationException("Support for continous outputs has not yet been" +
						" implemented.");
		}

		// Only create confusion matrixes if there is only 1 n value. Otherwise I would need to create a separate
		// set of confusion matrices for every value of n.
		if (nValues.size() == 1)
		{
			confusions = new ArrayList<>();
			for (int c = 0; c < metadata.cols(); c++)
			{
				confusions.add(new ConfusionMatrix(metadata.getAttrName(c)));
			}
		}

		correctCounts = new TreeMap<Integer, Integer>();
		for (Integer n : nValues)
			correctCounts.put(n, 0);
		
		totalCount = 0;
		
		this.metadata = new Matrix();
		this.metadata.copyMetadata(metadata);
	}

	@Override
	protected void evaluate(Vector target, List<Vector> predictions)
	{
		if (predictions.size() > largestN)
		{
			// Trim off unnecessary predictions.
			predictions = predictions.subList(0, largestN); 
		}

		for (Integer n : nValues)
		{
			List<Vector> predictionsSub;
			if (predictions.size() > n)
				 predictionsSub = predictions.subList(0, n);
			else
				predictionsSub = predictions;
			
			for (Vector pred : predictionsSub)
			{
				// If the target label is in predictionList, mark the prediction as correct.
				if (target.equals(pred))
				{
					correctCounts.put(n, correctCounts.get(n) + 1);
					break;
				}
			}
			
			if (nValues.size() == 1)
			{
				// Update confusion matrices.
				for (Vector pred : predictionsSub)
				{
					for (int lCol = 0; lCol < target.size(); lCol++)
					{
						// increment the count corresponding to the prediction.
						int targValue = (int) (Vector.isUnknown(target.get(lCol)) ? Double.MAX_VALUE : target.get(lCol));
						int predValue = (int) (Vector.isUnknown(pred.get(lCol)) ? Double.MAX_VALUE : pred.get(lCol));
						confusions.get(lCol).incrementCount(
								metadata.getAttrValueName(lCol, targValue), metadata.getAttrValueName(lCol, 
										predValue));
					}
				}
			}
		}
		
		totalCount++;

	}

	@Override
	protected List<Double> calcScores() 
	{
		if (totalCount == 0)
			throw new IllegalStateException("No instances have been evaluated.");
		
		List<Double> results = new ArrayList<>();
		for (int i : new Range(nValues.size()))
		{
			int n = nValues.get(i);
			results.add(((double)correctCounts.get(n)) / (totalCount));
		}
		return results;
	}

	@Override
	protected List<ConfusionMatrix> calcConfusions() 
	{
		return confusions;
	}

	@Override
	protected int getMaxDesiredSize()
	{
		return largestN;
	}

}















