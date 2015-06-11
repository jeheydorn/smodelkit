package smodelkit.learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.json.simple.JSONObject;

import smodelkit.Vector;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.util.Helper;
import smodelkit.util.Logger;

/**
 * A simple k-nearest neighbor learner.
 * @author joseph
 *
 */
public class KNN extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private int SORTING_THRESHOLD;
	private boolean USE_DISTANCE_WEIGHTS;
	private double LEAVE_ONE_OUT_REPS;
	private double LEAVE_ONE_OUT_THRESHOLD;
	private double MAX_WEIGHT = 1000000;
	private double DISTANCE_TO_UNKNOWN;
	private int k;

	// Training examples
	private Matrix tInputs, tLabels;
	
	/**
	 * @param randGenerator
	 * @param k Number of instances to use when predicting. 
	 * @param SORTING_THRESHOLD If k is greater than this, the closest k instances will be found by
	 * sorting the training set. When k is less than this, the ke instances will be found by choosing
	 * them one by one.
	 * @param USE_DISTANCE_WEIGHTS When true, the k instances will be wieghted by their distance to the
	 * query instance. When false, this weight is 1.
	 * @param LEAVE_ONE_OUT_REPS Perform leaveOneOut() this many times.
	 * @param LEAVE_ONE_OUT_THRESHOLD
	 * @param DISTANCE_TO_UNKNOWN Unknown features have this distance to any other feature, even other
	 * unknowns.
	 */
	public void configure(int k, int SORTING_THRESHOLD, boolean USE_DISTANCE_WEIGHTS,
			int LEAVE_ONE_OUT_REPS, double LEAVE_ONE_OUT_THRESHOLD, double DISTANCE_TO_UNKNOWN)
	{
		this.k = k;
		this.SORTING_THRESHOLD = SORTING_THRESHOLD;
		this.USE_DISTANCE_WEIGHTS = USE_DISTANCE_WEIGHTS;
		this.LEAVE_ONE_OUT_REPS = LEAVE_ONE_OUT_REPS;
		this.LEAVE_ONE_OUT_THRESHOLD = LEAVE_ONE_OUT_THRESHOLD;
		this.DISTANCE_TO_UNKNOWN = DISTANCE_TO_UNKNOWN;
	}
	
	@Override
	public void configure(JSONObject settings)
	{
		int k = (int)(long)(Long)settings.get("k");
		int sortingThreshold = (int)(long)(Long)settings.get("sorting_threshold");
		boolean useDistanceWeights = (Boolean)settings.get("use_distance_weights");
		int leaveOneOutReps = (int)(long)(Long)settings.get("leave_one_out_reps");
		double leaveOneOutThreshold = (Double)settings.get("leave_one_out_threshold");
		double distanceToUnknown = (Double)settings.get("distance_to_unknown");
		configure(k, sortingThreshold, useDistanceWeights, leaveOneOutReps,
				leaveOneOutThreshold, distanceToUnknown);
	}




	private double measureDistance(Vector input1, Vector input2)
	{
		assert(input1.size() == input2.size());
		assert(input1.size() == tInputs.cols());

		double difSum = 0;
		for(int i = 0; i < input1.size(); i++)
		{
			if (Vector.isUnknown(input1.get(i)) || Vector.isUnknown(input2.get(i)))
			{
				// I'm defining the distance to an unknown value to be a constant.
				difSum += DISTANCE_TO_UNKNOWN; 
			}
			else if (tInputs.getValueCount(i) == 0)
			{
				// The attribute is continuous.
				difSum += (input1.get(i) - input2.get(i)) * (input1.get(i) - input2.get(i));
			}
			else
			{
				// The attribute is nominal.
				difSum += input1.get(i) == input2.get(i) ? 0 : 1;
			}
		}

		return Math.sqrt(difSum);
	}

	/**
	 * 
	 * @param outInstances Must be size k.
	 * @param outLabels Must be size k.
	 */
	private void getKNearest(final Vector query, Vector[] outInstances, double[] outLabels)
	{
		assert outInstances.length == k;
		assert outLabels.length == k;

		// Create a list of indexes for sorting the instances and labels based on their distance from query
		// without changing the inputs or labels. 
		List<Integer> indexes = new ArrayList<Integer>(tInputs.rows());
		List<Integer> nearestIndexes = new ArrayList<Integer>(k);
		for(int i = 0; i < tInputs.rows(); i++)
			indexes.add(i);
		List<Integer> indexPtr = null;

		if (k >= SORTING_THRESHOLD)
		{
			// Sorting is faster for large k.
			
			Comparator<Integer> comparator = new Comparator<Integer>()
			{
				@Override
				public int compare(Integer i1, Integer i2)
				{
					double d1 = measureDistance(query, tInputs.row(i1));
					double d2 = measureDistance(query, tInputs.row(i2));
					return Double.compare(d1, d2);
				}
			};
			
			Collections.sort(indexes, comparator);
			indexPtr = indexes;
		}
		else
		{
			// Individual selection is faster for small k.
			for(int i = 0; i < k; i++)
			{
				nearestIndexes.add(getNearestInputIndex(query, indexes));
				indexes.remove(nearestIndexes.get(i));
			}
			indexPtr = nearestIndexes;
		}

		for(int i = 0; i < k; i++)
		{
			outInstances[i] = tInputs.row(indexPtr.get(i));
			outLabels[i] = tLabels.row(indexPtr.get(i)).get(0);
		}

	}


	private double calcWeight(Vector input1, Vector input2)
	{
		if (!USE_DISTANCE_WEIGHTS)
			return 1;

		double distance = measureDistance(input1, input2);
		if (distance == 0)
			return -1;
		return 1/(distance*distance);
	}

	private int getNearestInputIndex(Vector query, List<Integer> indexes)
	{
		int nearest = 0;
		double bestDistance = measureDistance(query, tInputs.row(indexes.get(0)));
		for(int i = 1; i < indexes.size(); i++)
		{
			double distance = measureDistance(query, tInputs.row(indexes.get(i)));
			if (distance < bestDistance)
			{
				bestDistance = distance;
				nearest = i;
			}
		}
		int nearestInput = indexes.get(nearest);
		return nearestInput;
	}


	// Reduces the data set while trying to preserve accuracy using the "leave one out" algorithm. 
	private void leaveOneOut()
	{
		tInputs.shuffle(rand, tLabels);
		Vector prediction;

		int originalRowCount = tInputs.rows();

		for(int i = 0; i < originalRowCount; i++)
		{
			if (tInputs.rows() <= k)
				break;

			Vector tempInput = tInputs.row(0);
			Vector tempLabel = tLabels.row(0);
			// These deleteRow statements will be very slow because Matrix is currently using an ArrayList.
			// If I really care about this code enough, I should make it more efficient.
			tInputs.removeRow(0);
			tLabels.removeRow(0);
			prediction = predict(tempInput, false);

			if (!tLabels.isContinuous(0) && prediction.get(0) == tempLabel.get(0)
			|| tLabels.isContinuous(0) && Math.abs(prediction.get(0) - tempLabel.get(0)) 
				< LEAVE_ONE_OUT_THRESHOLD)
			{
				// The temp data point was classified correctly, so leave it out of the training data.
			}
			else
			{
				// Put it back in.
				tInputs.addRow(tempInput);
				tLabels.addRow(tempLabel);
			}

			assert(tInputs.rows() == tLabels.rows());
			assert(tInputs.rows() <= originalRowCount);
		}

		Logger.println("Number of training instances after reduction: " + tInputs.rows() + "\n");
	}

	@SuppressWarnings("unused")
	private void randomDrop()
	{
		throw new UnsupportedOperationException();
//		tInstances.shuffle(randGenerator, tLabels);
//
//		while(tInstances.rows() > 96)
//		{
//			int delIndex = randGenerator.nextInt() % tInstances.rows();
//			tInstances.deleteRow(delIndex);
//			tLabels.deleteRow(delIndex);
//
//			assert(tInstances.rows() == tLabels.rows());
//			assert(tInstances.rows() >= k);
//		}
//
//		Logger.println("Number of training instances after reduction: " + tInstances.rows() + "\n");
	}

		// Train the model to predict the labels
	@Override
	public void innerTrain(Matrix inputs, Matrix labels)
	{
		assert inputs.rows() == labels.rows();
		assert k <= inputs.rows();
		assert labels.cols() == 1;

		String dwstr = USE_DISTANCE_WEIGHTS ? "on" : "off";
		Logger.println("Training knn, k: " + k + ", Distance weighting: " + dwstr + "\n");

		tInputs = inputs;
		tLabels = labels;

		for (int i = 0; i < LEAVE_ONE_OUT_REPS; i++)
			leaveOneOut();
	}

	@Override
	public Vector innerPredict(Vector input)
	{
		Vector prediction;
		assert input.size() == tInputs.cols();

		Vector[] kInputs = new Vector[k];
		double[] kLabels = new double[k];
		getKNearest(input, kInputs, kLabels);

		if (tLabels.isContinuous(0))
		{
			// Compute a weighted average of the k labels. The weights are based on input distance
			// if USE_DISTANCE_WEIGHTS = true.
			double weightSum = 0;
			double average = 0;
			for(int i = 0; i < k; i++)
			{
				double weight = calcWeight(input, kInputs[i]);
				weightSum += weight;
				average += weight * kLabels[i];
			}

			average /= weightSum;
			prediction = Vector.create(new double[]{average});
		}
		else
		{
			// The labels are discrete valued. 

			double[] totals = new double[tLabels.getValueCount(0)];
			for (int atrVal = 0; atrVal < tLabels.getValueCount(0); atrVal++)
			{
				for(int i = 0; i < k; i++)
				{
					if (kLabels[i] == atrVal)
					{
						double weight = calcWeight(input, kInputs[i]);
						if (weight == -1 || weight > MAX_WEIGHT)
						{
							// Exact match, or very close
							assert !Double.isNaN(weight);
							assert !Double.isInfinite(weight);
							if (weight > MAX_WEIGHT)
								Logger.println("A weight is larger than MAX_WEIGHT. It will be set to MAX_WEIGHT.");
							weight = MAX_WEIGHT;
						}
						totals[atrVal] += weight;
					}
				}
			}

			// Find the attribute with the largest weighted vote.
			// (Attribute values are integer numbers. Here the index into totals is the attribute value.)
			prediction = Vector.create(new double[]{ Helper.indexOfMaxElementInRange(totals, 0, totals.length)});
		}
		return prediction;
	}
	
	@Override
	protected boolean canImplicitlyHandleNominalFeatures()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleContinuousFeatures()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleNominalLabels()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleContinuousLabels()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleUnknownInputs()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleMultipleOutputs()
	{
		return false;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
		return false;
	}
	
	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return false;
	}

}
