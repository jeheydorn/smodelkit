package smodelkit.learner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.evaluator.TopN;
import smodelkit.filter.Filter;
import smodelkit.filter.Normalize;
import smodelkit.util.Helper;
import smodelkit.util.Logger;
import smodelkit.util.Range;

/**
 * A chain classifier which predicts a scored list of output vectors using a beam search.
 * @author joseph
 *
 */
public class RankedCC extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	SupervisedLearner[] models;
	boolean useAllPreviousPredictions;
	// An increased beamWidth allows RankedCC to return multiple outputs in a ranked list.
	int beamWidth;
	JSONObject submodelSettings;
	boolean useLabelsAsInputWhileTraining;
	boolean passOutputWeightsUnfiltered;
	/**
	 * Determines whether branch scores are added together or multiplied.
	 */
	String aggregationOperator;
	boolean useWrapper;
	Double wrapperValidationPercent;
	int wrapperFeatureSelectionReps;
	private String submodelName;
	
		
	/**
	 * @param submodelName The name of the sub-model to use in the chain.
	 * @param submodelSettings The settings for the sub-models in the chain.
	 * @param branchingFactor Only the to scoring branchingFactor number of branches will be considered for
	 * branching.
	 * @param useAllPreviousPredictions If true, then each model the in the chain will received as input
	 * all predictions from previous models. If false, then each will only received the prediction of the
	 * previous model.
	 * @param beamWidth The maximum size of the beam while predicting.
	 * @param useLabelsAsInputWhileTraining If true, then each model in the chain is trained with label values
	 * as inputs. If false, then each is trained with predictions from previous models in the chain.
	 * @param passOutputWeightsUnfiltered If true, then each model is trained and predicts with output weights
	 * from previous models rather than nominal values. If false, then each model trains and predicts with nominal
	 * values as inputs.
	 * @param aggregationOperator This can be "+" or "*". This is the operator used to collect scores while doing
	 * the beam search.
	 * @param useWrapper If true, then a FeatureWrapper will be used with each model in the chain. The wrapper
	 * removes inputs from previous models if they don't help improve results on a validation set. The original
	 * inputs are always used. The FeatureWrapper must train models multiple times, so this slows down training
	 * a lot, and doesn't seem to improve results.
	 * @param wrapperValidationPercent The percent of the training data to use for the FeatureWrapper's validation
	 * set if useWrapper==true. 
	 * @param wrapperFeatureSelectionReps See featureSelectionReps in the constructor of FeatureWrapper.java.
	 * 
	 */
	public void configure(String submodelName, JSONObject submodelSettings, boolean useAllPreviousPredictions,
			int beamWidth, boolean useLabelsAsInputWhileTraining,
			boolean passOutputWeightsUnfiltered, String aggregationOperator, boolean useWrapper, 
			Double wrapperValidationPercent, int wrapperFeatureSelectionReps)
	{
		this.submodelName = submodelName;
		this.submodelSettings = submodelSettings;
		this.useAllPreviousPredictions = useAllPreviousPredictions;
		this.beamWidth = beamWidth;
		this.useLabelsAsInputWhileTraining = useLabelsAsInputWhileTraining;
		this.passOutputWeightsUnfiltered = passOutputWeightsUnfiltered;
		this.aggregationOperator = aggregationOperator;
		this.useWrapper = useWrapper;
		this.wrapperValidationPercent = wrapperValidationPercent;
		this.wrapperFeatureSelectionReps = wrapperFeatureSelectionReps;
		
		if (!(aggregationOperator.equals("+") || aggregationOperator.equals("*")))
				throw new IllegalArgumentException();
		if (passOutputWeightsUnfiltered && useLabelsAsInputWhileTraining)
			throw new IllegalArgumentException("It doesn't make sense to not filter (to nominal values) output weights while training and"
					+ " predicting if I use only labels as inputs while training.");
		if (passOutputWeightsUnfiltered && beamWidth > 1)
			throw new UnsupportedOperationException("Branching currently doesn't work with passing output weights directly.");
		if (useWrapper && wrapperValidationPercent == null)
			throw new IllegalArgumentException("When useWrapper is true, wrapperValidationPercent must be specified.");
	}
	

	@Override
	public void configure(JSONObject settings)
	{
		boolean useAllPreviousPredictions = (boolean)settings.get("useAllPreviousPredictions");
		boolean useLabelsAsInputWhileTraining = (boolean)settings.get("useLabelsAsInputWhileTraining");
		boolean passOutputWeightsUnfiltered = (boolean)settings.get("passOutputWeightsUnfiltered");
		int beamWidth = (int)(long)(Long)settings.get("beam_width");
		String aggregationOperator = (String)settings.get("aggregation_operator");
		boolean useWrapper = (boolean)settings.get("useWrapper");
		Double wrapperValidationPercent = (Double)settings.get("wrapperValidationPercent");
		int wrapperFeatureSelectionReps = (int)(long)(Long)settings.get("wrapperFeatureSelectionReps");
		
		String submodelName = (String) settings.get("submodelName");
		String submodelSettingsFile = (String) settings.get("submodelSettingsFile");
		JSONObject submodelSettingsJson = MLSystemsManager.parseModelSettingsFile(submodelSettingsFile);
		configure(submodelName, submodelSettingsJson, 
				useAllPreviousPredictions,
				beamWidth, useLabelsAsInputWhileTraining, 
				passOutputWeightsUnfiltered, aggregationOperator, useWrapper, wrapperValidationPercent,
				wrapperFeatureSelectionReps);		
	}

	public void innerTrain(Matrix inputs, Matrix labels)
	{
		Logger.indent();
		// I need to be able to mutate inputs, so I am making a copy so I don't mess up something later by
		// changing the original.
		Matrix inputsMut = new Matrix(inputs);
		
		if (labels.cols() == 0)
			throw new IllegalArgumentException("Expected at least one column in labels.");
		
		Logger.print("Label column order: ");
		Logger.println(Helper.listToStringWithSeparator(
				new Range(labels.cols()).stream().map(c -> labels.getAttrName(c)).collect(Collectors.toList()), ", "));
		
		
		// Train MLPs M1 through Mn one at a time. After training each one transform the inputs.
		models = new SupervisedLearner[labels.cols()];
		assert(models.length > 0);	
		
		
		// Train M1 to predict X -> y1
		// Train M2 to predict X+y1 -> y2
		// 	.
		// 	.
		// 	.
		// Train Mn to predict X+(y1...yn-1) -> yn
		for (int m = 0; m < models.length; m++)
		{
			if (useWrapper)
			{
				List<Integer> featureColumnsToAlwaysUse = Helper.iteratorToList(new Range(inputs.cols()));
				FeatureWrapper fw = new FeatureWrapper(featureColumnsToAlwaysUse, submodelName, submodelSettings,
						new TopN(Arrays.asList(1)), wrapperValidationPercent, wrapperFeatureSelectionReps);
				fw.setRandom(rand);
				models[m] = fw;
			}
			else
			{
				models[m] = MLSystemsManager.createLearner(rand, submodelName, submodelSettings);
			}
			Matrix curLabel = labels.getColumns(m, 1);
			if (passOutputWeightsUnfiltered)
			{
				// Disable normalization of inputs from previous models.
				if (useWrapper)
				{
					throw new UnsupportedOperationException("If I want to use passOutputWeightsUnfiltered with useWrapper, I must devise a way "
							+ "to get it to work with the Normalize filter so that I do not normalize previouse output weights."
							+ " This currenlty doesn't work with useWrapper.");
				}
				Normalize nFilter = models[m].getFilter().findFilter(Normalize.class);
				for (int i : new Range(inputs.cols(), inputsMut.cols()))
				{
					nFilter.ignoreInputAttribute(i);
				}
			}
			models[m].train(inputsMut, curLabel);
		
				
			// Add the model's output to the inputs.
			if (m < models.length - 1)
			{
				if (!useAllPreviousPredictions)
				{
					inputsMut = new Matrix(inputs);
				}
				if (useLabelsAsInputWhileTraining)
				{
					// While  training, each NN in HBS uses as input the correct labels instead of the output
					// of previous NNs. 
					inputsMut.copyColumns(curLabel, 0, curLabel.cols());
				}
				else
				{
					// While training, each NN is HBS uses as input the predictions from previous NNs. 
					if (passOutputWeightsUnfiltered)
					{
						// Pass the output weights directly to the next model.
						Matrix predictions = new Matrix();
						for (int i : new Range(inputsMut.rows()))
						{
							double[] predFiltered = models[m].predictOutputWeights(inputsMut.row(i)).get(0);

							if (predictions.cols() == 0)
							{
								// This should only happen once in this loop.
								predictions.setSize(0, predFiltered.length);								
							}
							predictions.addRow(new Vector(predFiltered));
						}
						inputsMut.copyColumns(predictions, 0, predictions.cols());						
					}
					else
					{
						// Convert the output weights to nominal values before passing them to each model.
						Matrix predictions = new Matrix();
						predictions.copyMetadata(curLabel);
						for (int i : new Range(inputsMut.rows()))
						{
							predictions.addRow(models[m].predict(inputsMut.row(i)));
						}
						assert predictions.cols() == 1;
						inputsMut.copyColumns(predictions, 0, predictions.cols());
					}
				}
			}
			
		}
		Logger.unindent();
	}
	
	@Override
	protected Vector innerPredict(Vector input)
	{
//		Logger.printArray("Predicting input: ", input);
		
		List<Branch> branches = predictBeamSearch(input);

//		Logger.println("branches: " + branches);
		// Return the prediction of the branch with the highest score.
		Branch maxBranch = Collections.max(branches);
		double[] prediction = maxBranch.getAllPredictions();
//		Logger.printArray("prediction in innerPredict", prediction);
		return new Vector(prediction);
	}
		
	@Override
	protected List<Vector> innerPredictScoredList(Vector input, int maxDesiredSize)
	{
//		Logger.println("\npredicting input: " + trainInputs.rowToString(input));
		List<Branch> branches = predictBeamSearch(input);

			//		Logger.println("branches: ");
		List<Vector> predictions = new ArrayList<>();
		for (Branch b : branches)
		{
//			Logger.println(b);
			predictions.add(new Vector(b.getAllPredictions(), b.score));
		}
		
//		Logger.println("predictions:");
//		predictions.stream().forEach(tuple -> Logger.println(tuple.getFirst() + ", " + trainLabels.rowToString(tuple.getSecond())));
		
		return predictions;
	}

	@Override
	public List<double[]> innerPredictOutputWeights(Vector input)
	{		
		List<Branch> branches = predictBeamSearch(input);
		
		Branch maxBranch = Collections.max(branches);
		return maxBranch.getAllOutputWeightsInAList();
	}
	
	public List<Branch> predictBeamSearch(Vector input)
	{
		PriorityQueue<Branch> branches = new PriorityQueue<>();
		for (int m = 0; m < models.length; m++)
		{
			PriorityQueue<Branch> nextBranches = new PriorityQueue<>();
			if (branches.size() == 0)
			{
				double[] predFiltered = models[m].predictOutputWeights(input).get(0);
						 
				Filter modelFilter = useWrapper ? ((FeatureWrapper)models[m]).getInnerModelFilter() 
						: models[m].getFilter();
				addToBeam(nextBranches, generateBranches(predFiltered, modelFilter, null));
			}
			else
			{
				for (Branch branch : branches)
				{
					Vector curInput = null;
					if (passOutputWeightsUnfiltered)
					{		
						if (useAllPreviousPredictions)
						{
							curInput = input.concat(branch.getAllOutputWeights());
						}
						else
						{
							// Discard previous predictions
							curInput = input.concat(branch.outputWeights);
						}						
					}
					else
					{
						if (useAllPreviousPredictions)
						{
							curInput = input.concat(branch.getAllPredictions());
						}
						else
						{
							// Discard previous predictions
							curInput = input.concat(new double[]{branch.predNotFiltered});
						}
					}
							

					double[] outputWeights = models[m].predictOutputWeights(curInput).get(0);
					Filter modelFilter = useWrapper ? ((FeatureWrapper)models[m]).getInnerModelFilter() 
							: models[m].getFilter();
					addToBeam(nextBranches, generateBranches(outputWeights, modelFilter, branch));
				}
			}
			branches = nextBranches;
		}
		
		List<Branch> predictions = new ArrayList<>();
		while (!branches.isEmpty())
		{
			predictions.add(branches.remove());
		}
		
		Collections.reverse(predictions);
		
		return predictions;
	}


	/**
	 * Adds candidates to the beam, enforcing the beam width.
	 */
	private void addToBeam(PriorityQueue<Branch> beam, List<Branch> candidates)
	{
		// This could be more efficient if I inserted each branch into the beam in it's sorted position.
		beam.addAll(candidates);

		while (beam.size() > beamWidth)
		{
			beam.remove();
		}
	}

	private List<Branch> generateBranches(double[] outputWeights, Filter modelFilter, Branch prevBranch)
	{
		double prevScore;
		if (prevBranch == null)
		{
			if (aggregationOperator.equals("+"))
				prevScore = 0.0;
			else
				prevScore = 1.0;
		}
		else
		{
			prevScore = prevBranch.score;
		}
		
		assert outputWeights.length > 1;
		List<Branch> result = new ArrayList<Branch>();
		for (int i = 0; i < outputWeights.length; i++)
		{
			double newScore = aggregationOperator.equals("+") ? prevScore + outputWeights[i] :
				prevScore * outputWeights[i];
			result.add(new Branch((double) i, newScore, outputWeights, prevBranch));
		}
		return result;
	}
	
	private static class Branch implements Comparable<Branch>
	{
		/**
		 * The unFiltered prediction made by the NN.
		 */
		double predNotFiltered;
		/**
		 * The sum so far of the best scores in each branch before this branch.
		 */
		double score;
		/**
		 * A backpointer to allow me to get the full prediction for this branch.
		 */
		Branch prevBranch;
		/**
		 * The weights for each nominal value created by a sub-model.
		 */
		double[] outputWeights;
		
		double[] getAllPredictions()
		{
			if (prevBranch == null)
				return new double[] {predNotFiltered};
			else 
				return Helper.concatArrays(prevBranch.getAllPredictions(), new double[] {predNotFiltered});
		}
		
		List<double[]> getAllOutputWeightsInAList()
		{
			if (prevBranch == null)
			{
				List<double[]> result = new ArrayList<>();
				result.add(outputWeights);
				return result;
			}
			else 
			{
				List<double[]> prev = prevBranch.getAllOutputWeightsInAList();
				prev.add(outputWeights);
				return prev;
			}
		}

		double[] getAllOutputWeights()
		{
			if (prevBranch == null)
				return outputWeights;
			else 
				return Helper.concatArrays(prevBranch.getAllOutputWeights(), outputWeights);
		}

		public Branch(double predNotFiltered, double score, double[] outputWeights, Branch prevBranch)
		{
			this.predNotFiltered = predNotFiltered;
			this.score = score;
			this.prevBranch = prevBranch;
			this.outputWeights = outputWeights;
		}

		@Override
		public int compareTo(Branch other)
		{
			return Double.compare(this.score, other.score);
		}
		
		@Override
		public String toString()
		{
			return String.format("Branch: {score=%s, prediction=%s, getAllOutputWeights()=%s}", 
					Helper.formatDouble(score),
					Helper.arrayToString(getAllPredictions()), Helper.arrayToString(getAllOutputWeights()));
		};
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
};

