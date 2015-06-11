package smodelkit.learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.json.simple.JSONObject;

import smodelkit.Vector;
import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.util.Counter;
import smodelkit.util.Helper;
import smodelkit.util.Logger;
import smodelkit.util.Tuple2Comp;

/**
 * This is my implementation of Richard Morris's Hierarchical Multi-Output Nearest Neighbor Model.
 * HMONN uses an independent classifiers combined with a KNN to predict output vectors. I have
 * modified HMONN to predict multiple output vectors. 
 * @author joseph
 *
 */
public class HMONN extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private double DISTANCE_TO_UNKNOWN;
	private int k;
	private SupervisedLearner inn;
	private double theta;

	// Training examples
	private Matrix tInputs, tLabels;
	
	/**
	 * @param randGenerator
	 * @param k Number of instances to use when predicting. 
	 * @param DISTANCE_TO_UNKNOWN Unknown features have this distance to any other feature, even other
	 * unknowns.
	 * @param icSettings Settings for the IndependentClassifier model used to make initial predictions.
	 */
	public void configure(int k, double theta, double DISTANCE_TO_UNKNOWN, JSONObject icSettings)
	{
		this.k = k;
		this.theta = theta;
		this.DISTANCE_TO_UNKNOWN = DISTANCE_TO_UNKNOWN;
		this.inn = MLSystemsManager.createLearner(rand, "ic", icSettings);
	}
	
	@Override
	public void configure(JSONObject settings)
	{
		int k = (int)(long)(Long)settings.get("k");
		double theta = (Double)settings.get("theta");
		double distanceToUnknown = (Double)settings.get("distance_to_unknown");
		String icSettingsFile = (String)settings.get("ic_settings_file");
		JSONObject icSettings = MLSystemsManager.parseModelSettingsFile(icSettingsFile);
		configure(k, theta, distanceToUnknown, icSettings);
		
	}


	@Override
	public void innerTrain(Matrix inputs, Matrix labels)
	{
		assert inputs.rows() == labels.rows();
		assert k <= inputs.rows();
				
		Logger.indent();
		Logger.println("Training HMONN");
		Logger.println("DISTANCE_TO_UNKNOWN: " + DISTANCE_TO_UNKNOWN);
		Logger.println("k: " + k);
		tInputs = inputs;
		tLabels = labels;
		
		inn.train(inputs, labels);
		Logger.unindent();
	}

	@Override
	public Vector innerPredict(Vector input)
	{
		Logger.indent();
		assert input.size() == tInputs.cols();
		
		// First, predict using inn.
		Vector innerPred = inn.predict(input);
		// Combine inn's prediction with the input to make a query.
		Vector query = new VectorDouble(input);
		query.addAll(innerPred);

		
		// Find the k rows in the dataset which are closest to the query.
		Matrix kLabels = getKNearest(query);
		assert kLabels.rows() == k;
		
		// Find the most common label vector in kLabels.
		Vector prediction = findMostCommonLabel(kLabels);

		Logger.unindent();
		return prediction;
	}
	
	@Override
	protected List<Vector> innerPredictScoredList(Vector input, int maxDesiredSize)
	{
		assert input.size() == tInputs.cols();
		
		// First, predict using inn.
		Vector innerPred = inn.predict(input);
		// Combine inn's prediction with the input to make a query.
		Vector query = input.concat(innerPred);

		// Find the k rows in the dataset which are closest to the query.
		Matrix kLabels = getKNearest(query);
		assert kLabels.rows() == k;

		Counter<Vector> counts = new Counter<>();
		for (Vector label : kLabels)
		{
			counts.increment(label);
		}
		List<Vector> result = counts.toListFromHighToLow().stream().map(
				tuple -> new VectorDouble(tuple.getFirst(), (double)(int)tuple.getSecond()))
				.collect(Collectors.toList());
		return result;
	}

	private Vector findMostCommonLabel(Matrix kLabels)
	{
		Map<Vector, Integer> counts = new TreeMap<>();
		
		for (int r = 0; r < kLabels.rows(); r++)
		{
			Integer prevCount = counts.get(kLabels.row(r));
			if (prevCount == null)
				prevCount = 0;
			counts.put(kLabels.row(r), prevCount + 1);
		}
		return Helper.argmax(counts);
	}

	private Matrix getKNearest(final Vector query)
	{
		Matrix resultLabels = new Matrix();
		resultLabels.copyMetadata(tLabels);
		
		List<Integer> nearestIndexes = getKNearestInstanceIndexes(query);

		for(int i = 0; i < k; i++)
		{
			resultLabels.addRow(tLabels.row(nearestIndexes.get(i)));
		}
		
		return resultLabels;
	}
	
	
	/**
	 * Create a Queue of indexes to the closes k rows so far. Whith each comparison, call a method which either inserts 
	 * the new row and pops the worst one, or discards the new row.
	 * 
	 */

	private List<Integer> getKNearestInstanceIndexes(Vector query)
	{
		// Create a list of indexes for sorting the inputs and labels based on their distance from query
		// without changing the inputs or labels. 
		PriorityQueue<Tuple2Comp<Double, Integer>> queue = new PriorityQueue<Tuple2Comp<Double, Integer>>(k, new Comparator<Tuple2Comp<Double, Integer>>()
				{
					public int compare(Tuple2Comp<Double, Integer> o1,
							Tuple2Comp<Double, Integer> o2)
					{
						// Compare distance to query, where higher distances will be removed first in the queue.
						return -Double.compare(o1.getFirst(), o2.getFirst());
					}
				});

		if (tLabels.rows() < k)
			throw new IllegalArgumentException();
		// Load the first k rows into the k.
		for (int i = 0; i < k; i++)
		{
			double distance = measureDistance(query, tInputs.row(i).concat(tLabels.row(i)));
			queue.add(new Tuple2Comp<>(distance, i));
		}
		for (int i = k; i < tLabels.rows(); i++)
		{
			double distance = measureDistance(query, tInputs.row(i).concat(tLabels.row(i)));
			queue.add(new Tuple2Comp<>(distance, i));
			queue.remove();
		}
		
		List<Integer> result = new ArrayList<>();
		while (!queue.isEmpty())
		{
			result.add(queue.remove().getSecond());
		}
		Collections.reverse(result);
		
		return result;
	}
	
	private double measureDistance(Vector row1, Vector row2)
	{
		assert(row1.size() == row2.size());
		assert(row1.size() == tInputs.cols() + tLabels.cols());

		// theta=0 means only use predicted labels.
		// theta=1 means only use inputs (just do KNN).
		double difSum = 0;
		for(int i = 0; i < tInputs.cols(); i++)
		{
			difSum += theta * measureSingleDistance(row1, row2, i);
		}

		for(int i = 0; i < tLabels.cols(); i++)
		{
			difSum += (1.0 - theta) * measureSingleDistance(row1, row2, i + tInputs.cols());
		}

		return Math.sqrt(difSum);
	}
	
	private double measureSingleDistance(Vector row1, Vector row2, int index)
	{
		if (Vector.isUnknown(row1.get(index)) || Vector.isUnknown(row2.get(index)))
		{
			// I'm defining the distance to an unknown value to be a constant.
			// In my experiments as of 2/25/2014, this never gets called.
			throw new RuntimeException("To match Richard's setup, unkowns should have been filled.");
			//return DISTANCE_TO_UNKNOWN; 
		}
		boolean isContinuous = index < tInputs.cols() ? tInputs.isContinuous(index) 
				: tLabels.isContinuous(index - tInputs.cols());
		if (isContinuous)
		{
			double d = row1.get(index) - row2.get(index);
			return d * d;
		}
		else
		{
			// The attribute is nominal.
			return row1.get(index) == row2.get(index) ? 0 : 1;
		}
		
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
		return false;
	}
	
	@Override
	protected boolean canImplicitlyHandleUnknownInputs()
	{
		return true;
	}
	
	@Override
	protected boolean canImplicitlyHandleMultipleOutputs()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
		return false;
	}
	
	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}

}
