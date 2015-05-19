package smodelkit.learner;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.ArithmeticUtils;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.filter.ReorderOutputs;
import smodelkit.util.Helper;
import smodelkit.util.Logger;
import smodelkit.util.PermutationIterator;
import smodelkit.util.Range;
import smodelkit.util.Sample;
import smodelkit.util.SequenceIterator;
import smodelkit.util.Tuple2;
import smodelkit.util.Tuple2Iterator;

/**
 * An ensemble of MDC classifiers. The final result is a ranked list of output vectors,
 * where the rank of an output vectors are determined by their summed scores from 
 * sub-models.
 * @author joseph
 *
 */
public class WOVEnsemble extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	List<SupervisedLearner> submodels;
	List<Double> modelWeights;
	boolean doBagging;
	private boolean useModelWeights;
	private int numPredictionsWhenSettingModelWeights;
	private boolean doChainOrderExperiment;

	/**
	 * Creates an ensemble of MDC classifiers.
	 * @param size The number of base models to use.
	 * @param doBagging If true, then while training the datasets given to each sub-model will
	 * be re-sampled.
	 * @param submodelNamesAndSettings Contains pairs of sub-model names and json settings for each.
	 * @param doBagging If true, then bagging (using instance weights) will be performed on the training instances
	 * given to each sub-model.
	 * @param useModelWeights  If true, then the predictions of each sub-model will be weighted by it's performance
	 * on the training instances. The performance of each sub-model on the training set is measured by the sum of
	 * the probability mass that model gives to the target output vector for each instance. This setting is useful
	 * if some members of the ensemble are weaker on some datasets than they are on others. It allows WOVE
	 * to pay less attention to those weaker members. 
	 * @param numPredictionsWhenSettingModelWeights If useModelWeights is true, then this determines how many
	 * predicted output vectors to consider when calculating the weight of each sub-model.
	 * @param doChainOrderExperiment Use experimental chair order.
	 */
	public void configure(List<Tuple2<String, JSONObject>> submodelNamesAndSettings, boolean doBagging, 
			boolean useModelWeights, int numPredictionsWhenSettingModelWeights, boolean doChainOrderExperiment)
	{
		this.doBagging = doBagging;
		this.useModelWeights = useModelWeights;
		this.numPredictionsWhenSettingModelWeights = numPredictionsWhenSettingModelWeights;
		this.doChainOrderExperiment = doChainOrderExperiment;
		
		submodels = new ArrayList<>();
		for (Tuple2<String, JSONObject> tuple : submodelNamesAndSettings)
		{
			String submodelName = tuple.getFirst();
			JSONObject settings = tuple.getSecond();
			submodels.add(MLSystemsManager.createLearner(rand, submodelName, settings));
		}
		
		modelWeights = new ArrayList<>();
	}
	

	@Override
	public void configure(JSONObject settings)
	{
		JSONArray submodelSettingsJson = (JSONArray) settings.get("submodels");
		List<Tuple2<String, JSONObject>> subModelSettings = new ArrayList<>();
		for (Object subModelObj : submodelSettingsJson)
		{
			JSONObject subModelJson = (JSONObject)subModelObj;
			String name = subModelJson.get("name").toString().trim();
			String settingsFile = subModelJson.get("settingsFile").toString();
			JSONObject submodelSettings = MLSystemsManager.parseModelSettingsFile(settingsFile);
			subModelSettings.add(new Tuple2<>(name, submodelSettings));
		}

		boolean doBagging = (boolean)settings.get("doBagging");
		boolean useModelWeights = (boolean)settings.get("useModelWeights");
		int numPredictionsWhenSettingModelWeights = (int)(long)settings.get("numPredictionsWhenSettingModelWeights");
		boolean doChainOrderExperiment = (boolean) settings.get("doChainOrderExperiment");
		
		configure(subModelSettings, doBagging, useModelWeights, 
				numPredictionsWhenSettingModelWeights, doChainOrderExperiment);
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{		
		List<List<Integer>> outputOrders = doChainOrderExperiment ? generateOutputOrdersForEachSubmodel(labels)
				: null;
		
		for (int i : new Range(submodels.size()))
		{
			SupervisedLearner model = submodels.get(i);
			if (doBagging)
			{
				// Old way.
				// Matrix[] m = Sample.sampleWithReplacement(rand, inputs, labels, 1.0);
				
				// new way.
				Matrix[] m = Sample.sampleWithReplacementUsingInstanceWeights(rand, inputs, labels);
				
				model.train(m[0], m[1]);
			}
			else
			{
				model.train(inputs, labels);
			}
			if (useModelWeights)
			{	
				double weightSum = 0;
				for (int r : new Range(inputs.rows()))
				{
					List<Vector> pred 
							= model.predictScoredList(inputs.row(r), numPredictionsWhenSettingModelWeights);
					// If the target label is one of the predicted ones, add the weight given to it.
					for (Vector v : pred)
					{
						if (v.equals(labels.row(r)))
						{
							weightSum += v.getWeight();
							break;
						}
					}
				}
				modelWeights.add(weightSum);
			}
			else
			{
				modelWeights.add(1.0);
			}
			
			if (doChainOrderExperiment)
			{	
				// Reconfigure the ReorderOutputs filter of the model.
				model.getFilter().findFilter(ReorderOutputs.class).setOutputColumnOrder(outputOrders.get(i));
			}
		}
		
		// Normalize the model weights to make them more intuitive.
		double[] normalized = Helper.toDoubleArray(modelWeights);
		Helper.normalize(normalized);
		modelWeights = Helper.toDoubleList(normalized);
		
		Logger.println("modelWeights: " + Helper.formatDoubleList(modelWeights));
	}
	
	private List<List<Integer>> generateOutputOrdersForEachSubmodel(Matrix labels)
	{		
		List<List<Integer>> result = new ArrayList<>();
		
		long totalPerms = ArithmeticUtils.factorial(labels.cols());
		
		// If there are more submodels than possible permutations of the outputs, add copies of all possible
		// permutations until I cannot add another copy of them without having more vectors than submodels.
		while(submodels.size() - result.size() > totalPerms)
		{
			for (List<Integer> perm : new PermutationIterator<Integer>(new Range(labels.cols()).toList()))
				result.add(perm);
		}
		
		Collection<List<Integer>> remainder = Sample.samplePermutationsWithoutReplacement(rand, submodels.size() - result.size(), 
				labels.cols()); // TODO
				
		result.addAll(remainder);
		
		// TODO Add the backward version of each permutation if it hasn't been already added. Make sure the result is still equal to submodels.size() in size.
		
		return result;
	}

	@Override
	protected Vector innerPredict(Vector input)
	{
		Map<Vector, Double> scores = getScoreMap(input);
		return Helper.argmax(scores);
	}
	
	@Override
	public List<Vector> innerPredictScoredList(Vector input, int maxDesiredSize)
	{
		Map<Vector, Double> scores = getScoreMap(input);
		
		// Convert the map to a scored list.
		List<Vector> result = scores.entrySet().stream().map(
				entry -> new Vector(entry.getKey(), entry.getValue())).collect(Collectors.toList());
		result.sort((v1, v2) -> -Double.compare(v1.getWeight(), v2.getWeight()));
		return result;
	}
	
	/**
	 * Returns a map where keys are predictions made by the models in innerModels for the given input,
	 * and values are the summed scores those models gave to those predictions.
	 * @return
	 */
	private Map<Vector, Double> getScoreMap(Vector input)
	{
		Map<Vector, Double> scores = new TreeMap<>();
		for (Tuple2<SupervisedLearner, Double> modelAndScore : new Tuple2Iterator<>(submodels, modelWeights))
		{
			SupervisedLearner model = modelAndScore.getFirst();
			double modelScore = modelAndScore.getSecond();
			List<Vector> scoreList = model.predictScoredList(input, Integer.MAX_VALUE);
			for (Vector v : scoreList)
			{
				Double score = scores.get(v);
				if (score == null)
				{
					score = 0.0;
				}
				scores.put(v, score + (v.getWeight() * modelScore));
			}
		}
		
		return scores;
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










