package smodelkit.learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Logger;
import smodelkit.util.Range;
import smodelkit.util.SequenceIterator;
import smodelkit.util.ThreadCounter;

/**
 * This learner trains a separate SupervisedLearner for each output in the labels it is given.
 * @author joseph
 *
 */
public class IndependentClassifiers extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	SupervisedLearner[] models;
	private JSONObject submodelSettings;
	private String submodelName;

	private void configure(String submodelName, JSONObject submodelSettings)
	{
		this.submodelSettings = submodelSettings;
		this.submodelName = submodelName;
		if (submodelName == null)
			throw new IllegalArgumentException("submodelName cannot be null.");
	}
	
	@Override
	public void configure(JSONObject settings)
	{
		String submodelSettingsFilename = (String)settings.get("submodelSettingsFile");
		JSONObject submodelSettings = MLSystemsManager.parseModelSettingsFile(submodelSettingsFilename);
		String submodelName = (String)settings.get("submodelName");
		
		configure(submodelName, submodelSettings);
	}

	@Override
	protected void innerTrain(final Matrix inputs, Matrix labels)
	{			
		Logger.indent();
		Logger.println("Training IndependentClassifiers");
		// Make one learner per column
		models = new SupervisedLearner[labels.cols()];
		List<Future<?>> futures = new ArrayList<Future<?>>();
		int threadsReserved = ThreadCounter.reserveThreadCount(labels.cols());
		int numThreads = Math.max(1, threadsReserved);
		ExecutorService exService = Executors.newFixedThreadPool(numThreads);
		try
		{
			for (int m = 0; m < models.length; m++)
			{
				models[m] = MLSystemsManager.createLearner(rand, submodelName, submodelSettings);
								
				final Matrix labelsNext = new Matrix();
				labelsNext.copyColumns(labels, m, 1);
				final SupervisedLearner learner = models[m];
				
				Runnable runnable = new Runnable()
				{
					@Override
					public void run()
					{
						learner.train(inputs, labelsNext);
					}
				};
				
				futures.add(exService.submit(runnable));
			}
	
			for (int m = 0; m < models.length; m++)
			{
				try
				{
					futures.get(m).get();
					Logger.println("Retrieved results for sub-model " + m + ".");
				}
				catch(ExecutionException e)
				{
					throw new RuntimeException(e);
				}
				catch(InterruptedException e)
				{
					throw new RuntimeException(e);
				}
			}
		}
		finally
		{
			exService.shutdown();
			ThreadCounter.freeThreadCount(threadsReserved);
		}
		Logger.unindent();
		Logger.println("Done training IndependentClassifiers");
	}

	@Override
	public Vector innerPredict(Vector input)
	{		
		Vector result = new Vector(new double[0]);
		for (SupervisedLearner learner : models)
		{
			result = result.concat(learner.predict(input));
		}
		return result;
	}
	
	@Override
	protected List<double[]> innerPredictOutputWeights(Vector input)
	{
		List<double[]> result = new ArrayList<>();
		for (SupervisedLearner learner : models)
		{
			List<double[]> learnerPred = learner.predictOutputWeights(input);
			assert learnerPred.size() == 1;
			result.add(learnerPred.get(0));
		}
		return result;
	}
	
	/**
	 * Does a best-first search from the highest scoring prediction to the lowest (or until maxDesiredSize
	 * is reached).
	 * The big-O is O(maxDesiredSize^2 * numOutputColumns * numOutputValues log(maxDesiredSize), where 
	 * numOutputColumns is the number of outputs in
	 * the dataset, and numOutputValues is the number of class values per output (assuming all output have the
	 * same number of classes). If maxDesiredSize is much smaller than the total number of possible output vectors, 
	 * (which it usually will be) then this is much more efficient than searching through all output vectors
	 * because there are an exponential number of them.
	 * 
	 * @return Predictions are returned in order from highest score to lowest. No more than maxDesiredSize
	 * predictions will be returned.
	 */
	@Override
	protected List<Vector> innerPredictScoredList(Vector input, int maxDesiredSize)
	{
		return doSearchForScoredList(input, maxDesiredSize);
		// I use the line below to generate unit test expected values for doSearchForScoredList.
		//return predictExaustive(input, maxDesiredSize);
	}
	
	private List<Vector> doSearchForScoredList(Vector input, int maxDesiredSize)
	{
		Set<Vector> explored = new TreeSet<>();

		List<double[]> weights = innerPredictOutputWeights(input);
		
		Vector initialPrediction = innerPredict(input);
		
		List<Vector> scoredPredictions = new ArrayList<>();
		
		explored.add(initialPrediction);
		scoredPredictions.add(new Vector(initialPrediction, getScoreForPrediction(weights, initialPrediction)));
		
		for (@SuppressWarnings("unused") int ignored : new Range(maxDesiredSize))
		{
			// Find the next best prediction that is not in explored. The score of a prediction is simply the
			// sum of the weights the sub-models gave to the prediction.
			Vector bestPred = null;
			double bestScore = Double.NEGATIVE_INFINITY;
	
			for (Vector prevPrediction : explored)
			{
				for (int c : new Range(weights.size())) // weights.size() is the number of output dimensions.
				{
					for (int outptuValue : new Range(weights.get(c).length))
					{
						Vector pred = new Vector(prevPrediction);
						pred.set(c, outptuValue);
						if (!explored.contains(pred))
						{
							double score = getScoreForPrediction(weights, pred);
							if (score >= bestScore)
							{
								bestPred = pred;
								bestScore = score;
							}
						}
					}
				}
			}
			
			if (bestPred == null)
			{
				// There are no more predictions to make
				break;
			}
			else
			{
				explored.add(bestPred);
				scoredPredictions.add(new Vector(bestPred, bestScore));
			}
		}		

		return scoredPredictions;
	}
	
	private double getScoreForPrediction(List<double[]> weights, Vector prediction)
	{
		double score = 1.0;
		for (int c : new Range(weights.size()))
		{
			// Make sure predictions are nominal.
			assert (int)prediction.get(c) == prediction.get(c);
			
			if (weights.get(c)[(int)prediction.get(c)] < 0)
				throw new IllegalArgumentException("Predicted weights must be between 0 and 1 inclusive.");
			if (weights.get(c)[(int)prediction.get(c)] > 1.0)
				throw new IllegalArgumentException("Predicted weights must be between 0 and 1 inclusive.");
			
			score *= weights.get(c)[(int)prediction.get(c)];
		}
		return score;
	}
	
	/**
	 * Predicts by trying every possible output vector. I only keep this method around to test
	 * doSearchForScoredList.
	 */
	@SuppressWarnings("unused")
	private List<Vector> predictExaustive(Vector input, int maxDesiredSize)
	{
		PriorityQueue<Vector> queue = new PriorityQueue<>(maxDesiredSize, 
				new Comparator<Vector>()
				{

					@Override
					public int compare(Vector o1, Vector o2)
					{
						// Keep the tuples with the highest scores.
						return Double.compare(o1.getWeight(), o2.getWeight());
					}
				});

		List<double[]> weights = innerPredictOutputWeights(input);
		List<Integer> outputRanges = new ArrayList<>();
		for (int c : new Range(weights.size()))
		{
			outputRanges.add(weights.get(c).length);
		}

		for (List<Integer> labelList : new SequenceIterator(outputRanges))
		{	
			Vector pred = convertListOfOutputValuesToVector(labelList);
			Vector candidate = new Vector(pred, getScoreForPrediction(weights, pred));
//			Logger.println("\ncandidate: " + candidate);
			queue.add(candidate);
			while(queue.size() > maxDesiredSize)
				queue.remove();
		}	
		
		// Extract the predictions from the queue.
		List<Vector> predictions = new ArrayList<>();
		while (!queue.isEmpty())
		{
			Vector tuple = queue.remove();
			predictions.add(tuple);
		}
		
		Collections.reverse(predictions);
		
		return predictions;
	}
	
	private Vector convertListOfOutputValuesToVector(List<Integer> labelList)
	{
		// Convert the label from list format to Vector.
		double[] label = new double[labelList.size()];
		for (int columnIndex : new Range(label.length))
		{
			label[columnIndex] = labelList.get(columnIndex);
		}
		return new Vector(label);
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
		// The code I use to make a scored list of predictions only works
		// if all outputs are nominal.
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
		return true;
	}
	
	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}

}
