package smodelkit.evaluator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import smodelkit.ConfusionMatrix;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Range;
import smodelkit.util.Tuple2;

/**
 * Similar to TopN except rather than searching for a predicted output vector that matches
 * the target exactly, this find the one that is the closes match according to hamming
 * distance, and counts the prediction (possibly) partially correct according to that
 * hamming distance.
 * 
 * When n=1 this is just the hamming score (which is 1 - (hamming distance)). 
 * 
 * Confusion matrices will be created while evaluating iff only 1 value of n is used.
 * @author joseph
 *
 */
public class TopNHamming extends Evaluator
{	
	private static final long serialVersionUID = 1L;
	private List<Integer> nValues;
	private ArrayList<ConfusionMatrix> confusions;
	private List<Double> correctCounts;
	private int largestN;
	Matrix metadata;
	private int totalCount;

	public TopNHamming()
	{
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
		
		largestN = Collections.max(nValues);
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
		
	private  <T> List<T> subListOrLess(List<T> items, int size)
	{
		if (items.size() <= size)
			return items;
		else
			return items.subList(0, size);
	}
	
	private int countNumCorrect(Vector d1, Vector d2)
	{
		int dist = 0;
		for (int c = 0; c < d1.size(); c++)
		{
			if (d1.get(c) == d2.get(c))
				dist++;
		}
		return dist;
	}
	
	@Override
	protected void startBatch(Matrix metadata)
	{
		// Make sure we don't have continuous labels.
		for (int i = 0; i < metadata.cols(); i++)
		{
			if (metadata.isContinuous(i))
				throw new IllegalArgumentException("HammingAccuracy does not work on continous labels.");
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

		correctCounts = new ArrayList<>();
		for (@SuppressWarnings("unused") int n : nValues)
			correctCounts.add(0.0);
		totalCount = 0;
		
		this.metadata = new Matrix();
		this.metadata.copyMetadata(metadata);


	}

	@Override
	protected void evaluate(Vector target, List<Vector> predictions)
	{
		for (int nIndex : new Range(nValues.size()))
		{
			int n = nValues.get(nIndex);
			// Get the n highest scored predictions and remove the scores.
			// Find the prediction which is closest to the target.
			Vector closestPrediction = subListOrLess(predictions, n).stream().map(
					pred -> new Tuple2<Vector, Integer>(pred, countNumCorrect(pred, target)
							)).max((t1, t2) -> Integer.compare(t1.getSecond(), t2.getSecond()))
							.get().getFirst();
			
			assert target.size() == closestPrediction.size();

			if (nValues.size() == 1)
			{
				// Update confusion matrices.
				for (int lCol = 0; lCol < target.size(); lCol++)
				{
					// increment the count corresponding to the prediction.
					int targValue = (int) (Vector.isUnknown(target.get(lCol)) ? Double.MAX_VALUE : target.get(lCol));
					int predValue = (int) (Vector.isUnknown(closestPrediction.get(lCol)) ? Double.MAX_VALUE : closestPrediction.get(lCol));
					confusions.get(lCol).incrementCount(
							metadata.getAttrValueName(lCol, targValue), metadata.getAttrValueName(lCol, 
									predValue));
				}
			}

			int correctCount = countNumCorrect(closestPrediction, target);
			correctCounts.set(nIndex, correctCounts.get(nIndex) + correctCount);
		
		}
		
		totalCount++;
	}

	@Override
	protected List<Double> calcScores()
	{
		// Divide each correct count by the number of output values to get an average over output values.
		List<Double> dividedCounts = new ArrayList<>(correctCounts.size());
		for (int i : new Range(correctCounts.size()))
			dividedCounts.add(correctCounts.get(i) / (totalCount * metadata.cols()));
		
		return dividedCounts;
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

	@Override
	public boolean higherScoresAreBetter()
	{
		return true;
	}


};
