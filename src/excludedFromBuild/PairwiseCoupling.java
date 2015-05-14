package smodelkit.learner;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.test.learners.OutputWeightsMockLearner;
import smodelkit.util.Helper;
import smodelkit.util.Pair;
import smodelkit.util.Range;

/**
 * A method for doing multi-class (> 2) classification with a classifier which
 * gives probabilities for binary classification. This is done by transforming the
 * dataset such that the labels contain a binary nominal attribute for each pair of 
 * nominal values in the original dataset. Pairwise coupling is used to convert
 * predictions of those nominal values (as probabilities) to a catagorical distribution
 * over values of the original nominal attribute. For more information on pairwise coupling, see: 
 * Hastie, Trevor, and Robert Tibshirani. "Classification by pairwise coupling." The annals of statistics 26.2 (1998): 451-471.
 *  
 */

public class PairwiseCoupling extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private SupervisedLearner[][] submodels;
	private JSONObject submodelSettings;
	private String submodelName;
	// This is an implementation of n as described in Hastie, Trevor, and Robert Tibshirani. "Classification by pairwise coupling." The annals of statistics 26.2 (1998): 451-471.
	private double[][] n;
	
	public PairwiseCoupling()
	{
	}
	
	public PairwiseCoupling(String submodelName, JSONObject submodelSettings)
	{
		configure(submodelName, submodelSettings);
	}
	
	private void configure(String submodelName, JSONObject submodelSettings)
	{
		this.submodelName = submodelName;
		this.submodelSettings = submodelSettings;
	}

	@Override
	public void configure(JSONObject settings)
	{
		String submodelName = (String)settings.get("submodelName");
		String submodelSettingsFile = settings.get("submodelSettingsFile").toString();
		JSONObject submodelSettings = MLSystemsManager.parseModelSettingsFile(submodelSettingsFile);
		configure(submodelName, submodelSettings);
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{
		if (labels.cols() != 1)
			throw new IllegalArgumentException("Expected only one label column, but got " + labels.cols());
		
		if (labels.getValueCount(0) <= 2)
		{
			// There is nothing to do if the task is binary classification or regression.
			submodels = new SupervisedLearner[][] {{ 
				MLSystemsManager.createLearner(rand, submodelName, submodelSettings)}};
			submodels[0][0].train(inputs, labels);
			return;
		}
		
		// A subset is a dataset to train a learner to do pairwise comparisons. All subsets have the same
		// structure. subDataLabelMeta is the structure of a label for a subset, which is just a single
		// binary value.
		Matrix subDataLabelMeta = new Matrix();
		//Create the label column.
		subDataLabelMeta.addEmptyColumn("class");
		subDataLabelMeta.addAttributeValue(subDataLabelMeta.cols() - 1, "i");
		subDataLabelMeta.addAttributeValue(subDataLabelMeta.cols() - 1, "j");
		
		submodels = new SupervisedLearner[labels.getValueCount(0)][labels.getValueCount(0)];
		n = new double[labels.getValueCount(0)][labels.getValueCount(0)];
		
		for (int i : new Range(labels.getValueCount(0)))
		{
			for (int j : new Range(i + 1, labels.getValueCount(0)))
			{
				// Prepare a dataset to train submodel[i][j] to predict the probability that
				// an instance has value i rather than j.
				
				Matrix subInputs = new Matrix();
				subInputs.copyMetadata(inputs);
		
				Matrix subLabels = new Matrix();
				subLabels.copyMetadata(subDataLabelMeta);
		
				for (int rowI : new Range(inputs.rows()))
				{
					if (labels.row(rowI).get(0) == i)
					{
						subInputs.addRow(inputs.row(rowI));
						subLabels.addRow(new Vector(new double[] {0.0}, inputs.row(rowI).getWeight()));
						n[i][j] += inputs.row(rowI).getWeight();
					}
					else if (labels.row(rowI).get(0) == j)
					{
						subInputs.addRow(inputs.row(rowI));
						subLabels.addRow(new Vector(new double[] {1.0}, inputs.row(rowI).getWeight()));
						n[i][j] +=inputs.row(rowI).getWeight();
					}
					// If instance r does not have label value i or j, then this sub-model doesn't use
					// that instance.
				}	
				
				submodels[i][j] = MLSystemsManager.createLearner(rand, submodelName, submodelSettings);
				submodels[i][j].train(subInputs, subLabels);
			}
		}

		// n is symmetric.
		for (int i : new Range(labels.getValueCount(0)))
		{
			for (int j : new Range(i + 1, labels.getValueCount(0)))
			{
				n[j][i] = n[i][j];
			}
		}
		
		// Clean up.
		submodelName = null;
		submodelSettings = null;
	}

	@Override
	protected Vector innerPredict(Vector input)
	{
		return new Vector(new double[] {Helper.indexOfMaxElement(innerPredictOutputWeights(input).get(0))});
	}
	
	@Override
	protected List<double[]> innerPredictOutputWeights(Vector input)
	{
		if (submodels.length == 1 && submodels[0].length == 1)
		{
			// Pairwise coupling cannot be applied if the task is binary classification or regression.
			return submodels[0][0].predictOutputWeights(input);
		}
		
		double[][] r = new double[submodels.length][submodels[0].length];
		
		for (int i : new Range(submodels.length))
		{
			for (int j : new Range(i + 1, submodels[i].length))
			{
				// r[i][j] is the probability that the label value is i rather than j.
				double[] weights = submodels[i][j].predictOutputWeights(input).get(0);
				assert weights.length == 2;
				// Make sure weights is normalized.
				double total = weights[0] + weights[1];
				double iProb = weights[0]/total;
				r[i][j] = iProb;
				r[j][i] = 1.0 - iProb;
			}
		}

		double[] probabilities = weka.classifiers.meta.MultiClassClassifier.pairwiseCoupling(n, r);
		return Collections.singletonList(probabilities);
	}

	public static void testPrivateMethods()
	{
		
		// Test equation 1.1 from:
		// Hastie, Trevor, and Robert Tibshirani. "Classification by pairwise coupling." The annals of statistics 26.2 (1998): 451-471.
		OutputWeightsMockLearner aVSb = new OutputWeightsMockLearner(
				Arrays.asList(Collections.singletonList(new double[] {0.9, 0.1})));
		OutputWeightsMockLearner aVSc = new OutputWeightsMockLearner(
				Arrays.asList(Collections.singletonList(new double[] {0.4, 0.6})));
		OutputWeightsMockLearner bVSc = new OutputWeightsMockLearner(
				Arrays.asList(Collections.singletonList(new double[] {0.7, 0.3})));
		
		PairwiseCoupling target = new PairwiseCoupling("zeror", 
				MLSystemsManager.parseModelSettingsFile("model_settings/zeror.json"));
		target.setRandom(new Random(0));
		
		// To prevent ZeroR from throwing an exception, I must create a dummy instance of every class value.
		Matrix data = new Matrix();
		data.loadFromArffString(
				"@relation test\n"
				+ "@attribute ignoredInput real\n"
				+ "@attribute class {a,b,c}\n"
				+ "@data\n"
				+ "99, a\n"
				+ "99, b\n"
				+ "99, c\n"
				+ "");
		
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();

		target.train(inputs, labels);
		// Change out the sub-models for mock objects. This allows me to test specific values for the
		// probabilities from the sub-models.
		target.submodels = new SupervisedLearner[][] 
				{{null, aVSb, aVSc},
					{null, null, bVSc},
					{null, null, null}};

		List<double[]> prediction = target.predictOutputWeights(inputs.row(0));
		assertEquals(1, prediction.size());
		assertArrayEquals(new double[] {0.47, 0.25, 0.28}, prediction.get(0), 0.1);
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
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
		return false;
	}

	@Override
	protected boolean canImplicitlyHandleMultipleOutputs()
	{
		return false;
	}

	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}

}
