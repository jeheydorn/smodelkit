package smodelkit.learner;

import static smodelkit.Vector.assertVectorEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
/**
 * An ensemble of MDC classifiers which selects each output as the one which the base models
 *  give the most weight to. Weights are scores given by base models.
 *  
 *  This is designed to work like Jesse Read's ensemble in "Multi-Dimensional Classification 
 *  with Super-Classes", section 5.
 */
import smodelkit.util.Range;
import smodelkit.util.Sample;

/**
 * An implementation of the ensemble technique used by Read et al. in their paper
 * "Multi-Dimensional Classification with Super-Classes".
 * @author joseph
 *
 */
public class MaxWeightEnsemble extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	List<SupervisedLearner> innerModels;
	boolean doBagging;

	/**
	 * Creates an ensemble of MDC classifiers.
	 * @param size The number of base models to use.
	 * @param doBagging If true, then while training the datasets given to each sub-model will
	 * be re-sampled.
	 */
	public void configure(String innerModelName, JSONObject innerModelSettings, int size,
			boolean doBagging)
	{
		this.doBagging = doBagging;
		
		innerModels = new ArrayList<>();
		for (@SuppressWarnings("unused") int i : new Range(size))
		{
			innerModels.add(MLSystemsManager.createLearner(rand, innerModelName, innerModelSettings));
		}
	}


	@Override
	public void configure(JSONObject settings)
	{
		String submodelName = (String)settings.get("submodelName");
		String submodelSettingsFile = settings.get("submodelSettingsFile").toString();
		JSONObject submodelSettings = MLSystemsManager.parseModelSettingsFile(submodelSettingsFile);
		int size = (int)(long)(Long)settings.get("size");
		boolean doBagging = (boolean)settings.get("doBagging");
		
		configure(submodelName, submodelSettings, size, doBagging);
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{		
		for (SupervisedLearner model : innerModels)
		{
			if (doBagging)
			{
				Matrix[] m = Sample.sampleWithReplacementUsingInstanceWeights(rand, inputs, labels);
				model.train(m[0], m[1]);
			}
			else
			{
				model.train(inputs, labels);
			}
		}
	}

	@Override
	public Vector innerPredict(Vector input)
	{
		// Each element in the outer list corresponds to a model in this ensemble. 
		// Each element is a list of weight vectors, where the n'th element of the
		// list corresponds to the n'th dimension. The k'th element of each weight
		// vector corresponds is a weight for the k'th output value.
		List<List<double[]>> outputWeightsPerModel = innerModels.stream().map(model 
				-> model.predictOutputWeights(input)).collect(Collectors.toList());
		
		
		// Reorganize outputWeightsPerModel so that the outer list is per dimension.
		int numLabelColumns = outputWeightsPerModel.get(0).size();
		List<List<double[]>> outputWeightsPerDim = new ArrayList<>(numLabelColumns);
		for (int dim : new Range(numLabelColumns))
		{
			List<double[]> dimWeights = new ArrayList<>();
			for (int m : new Range(outputWeightsPerModel.size()))
			{
				dimWeights.add(outputWeightsPerModel.get(m).get(dim));
			}
			outputWeightsPerDim.add(dimWeights);
		}


		return findLabelWithMostWeightPerColumn(outputWeightsPerDim);
	}
	
	private static Vector findLabelWithMostWeightPerColumn(List<List<double[]>> outputWeightsPerDim)
	{
		double[] result = new double[outputWeightsPerDim.size()];

		// For each output.
		for (int j : new Range(result.length))
		{
			
			int bestV = -1;
			double bestSum = -1;
			
			for (int v : new Range(outputWeightsPerDim.get(j).get(0).length))
			{
				double sum = 0;
				for (int m : new Range(outputWeightsPerDim.get(j).size()))
				{
					// Choose the output value with the maximum summed weight for that output from each model.
					sum += outputWeightsPerDim.get(j).get(m)[v];
				}
				
				if (sum > bestSum || bestV == -1)
				{
					bestSum = sum;
					bestV = v;
				}
				
			}
			
			result[j] = bestV;
		}
		return new VectorDouble(result);
					
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
	
	
	public static void testPrivateMethods()
	{
		// Test findLabelWithMostWeightPerColumn.
		{
			List<List<double[]>> outputWeights = Arrays.asList(
					Arrays.asList(
							new double[] {0.1, 0.1, 0.1},
							new double[] {0.9, 0.1, 0.1},
							new double[] {0.6, 0.4, 0.1},
							new double[] {0.5, 0.4, 0.1}));
			Vector actual = findLabelWithMostWeightPerColumn(outputWeights);
			assertVectorEquals(new VectorDouble(new double[] {0}), new VectorDouble(actual), 0.0000001);
		}
		
		{
			List<List<double[]>> outputWeights = Arrays.asList(
					Arrays.asList(
							new double[] {0.1, 0.1, 0.1},
							new double[] {0.2, 0.1, 0.1},
							new double[] {0.6, 0.4, 1.0},
							new double[] {0.2, 0.4, 1.0}));
			Vector actual = findLabelWithMostWeightPerColumn(outputWeights);
			assertVectorEquals(new VectorDouble(new double[] {2}), new VectorDouble(actual), 0.000001);
		}

		{
			List<List<double[]>> outputWeights = Arrays.asList(
					Arrays.asList(
							new double[] {0.1, 0.1, 0.1},
							new double[] {0.2, 0.1, 0.1},
							new double[] {0.6, 0.9, 0.0},
							new double[] {0.2, 0.9, 1.0}),
					Arrays.asList(
							new double[] {0.1, 0.1},
							new double[] {0.2, 0.1},
							new double[] {0.1, 0.9},
							new double[] {0.2, 0.4}
							));
			Vector actual = findLabelWithMostWeightPerColumn(outputWeights);
			assertVectorEquals(new VectorDouble(new double[] {1, 1}), actual, 0.000001);
		}

		{
			List<List<double[]>> outputWeights = Arrays.asList(
					Arrays.asList(
							new double[] {0.1, 0.1, 0.1},
							new double[] {0.2, 0.1, 0.1},
							new double[] {0.9, 0.0, 0.0},
							new double[] {1.0, 0.6, 1.0}),
					Arrays.asList(
							new double[] {0.1, 0.1},
							new double[] {0.2, 0.1},
							new double[] {0.1, 0.9},
							new double[] {0.2, 0.4}
							));
			Vector actual = findLabelWithMostWeightPerColumn(outputWeights);
			assertVectorEquals(new VectorDouble(new double[] {0, 1}), actual, 0.000001);
		}

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



