package smodelkit.learner;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.evaluator.Evaluator;
import smodelkit.evaluator.TopN;
import smodelkit.filter.Filter;
import smodelkit.util.Helper;
import smodelkit.util.Logger;
import smodelkit.util.Range;
import smodelkit.util.Tuple2;

/**
 * A wrapper for doing feature selection on select input columns.
 * @author joseph
 *
 */
public class FeatureWrapper extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	List<Integer> featureColumnsToAlwaysUse;
	String innerModelName;
	JSONObject innerModelSettings;
	SupervisedLearner innerModel;
	List<Integer> innerModelFeatureColumns;
	Evaluator evaluator;
	Double validationSetPercent;
	int featureSelectionReps;
	
	@Override
	public void configure(JSONObject settings)
	{
		// I never create FeatureWrapper from the command line.
		throw new UnsupportedOperationException();
	}

	/**
	 * 
	 * @param featureColumnsToAlwaysUse Columns specified by this parameter will always be retained as input
	 * during feature selection.
	 * @param evaluator  When determining which feature columns to retain, this will be used to create scores.
	 * @param featureSelectionReps This can b 1 or > 1. 1 means apply the FeatureWrapper one time and keep
	 * each input if it improves validation set results. Values > 1 mean apply it multiple times and only keep
	 * an input if it improves validation set results every time.
	 */
	public FeatureWrapper(List<Integer> featureColumnsToAlwaysUse, String innerModelName,
			JSONObject innerModelSettings, Evaluator evaluator, Double validationSetPercent,
			int featureSelectionReps)
	{
		this.featureColumnsToAlwaysUse = featureColumnsToAlwaysUse;
		this.innerModelName = innerModelName;
		this.innerModelSettings = innerModelSettings;
		this.evaluator = evaluator;
		this.validationSetPercent = validationSetPercent;
		this.featureSelectionReps = featureSelectionReps;
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{
		Logger.indent();
		
		Logger.println("Predicting column: " + inputs.cols());
		
		if (featureColumnsToAlwaysUse.size() == inputs.cols())
		{
			innerModel = trainOnColumns(inputs, labels, featureColumnsToAlwaysUse);
			innerModelFeatureColumns = featureColumnsToAlwaysUse;			
			Logger.println("Columns kept: " + innerModelFeatureColumns);
			Logger.unindent();
			return;
		}
		
		Logger.println("featureSelectionReps: " + featureSelectionReps);
		List<List<Integer>> keepList = new ArrayList<>();
		for (@SuppressWarnings("unused") int i : new Range(featureSelectionReps))
		{
			keepList.add(chooseInnerModelFeatureColumns(inputs, labels));
		}
		List<Integer> keepColumns = new ArrayList<>();
		for (int c : new Range(inputs.cols()))
		{
			if (keepList.stream().filter(k -> k.contains(c)).count() == featureSelectionReps)
			{
				keepColumns.add(c);
			}
		}
		
		// Retrain a new model with all instances.
		innerModelFeatureColumns = keepColumns;
		innerModel = trainOnColumns(inputs, labels, keepColumns);
		
		Logger.println("Columns kept: " + innerModelFeatureColumns);
		Logger.unindent();
	}
	
	private List<Integer> chooseInnerModelFeatureColumns(Matrix inputs, Matrix labels)
	{
		// I'm copying these so that I don't shuffle the original inputs and labels.
		Matrix inputsShuffled = new Matrix(inputs);
		Matrix labelsShuffled = new Matrix(labels);
		inputs = null;
		labels = null;
		inputsShuffled.shuffle(rand, labelsShuffled);

		Matrix[] sets = createValidationSet(inputsShuffled, labelsShuffled, validationSetPercent);
		Matrix tInputs = sets[0];
		Matrix tLabels = sets[1];
		Matrix vInputs = sets[2];
		Matrix vLabels = sets[3];

		// Greedily search for the best columns to train model i on.
			
		List<Integer> keepColumns = new ArrayList<>(featureColumnsToAlwaysUse);
		
		double lastIterationScore;
		if (featureColumnsToAlwaysUse.isEmpty())
		{
			lastIterationScore = Double.NEGATIVE_INFINITY;
		}
		else
		{
			SupervisedLearner initialModel = trainOnColumns(tInputs, tLabels, keepColumns);
			lastIterationScore = validateOnColumns(vInputs, vLabels, keepColumns, initialModel);
		}
		
		Logger.println("initial lastIterationScore: " + lastIterationScore);
		
		List<Integer> featureSelectionCols = Helper.iteratorToList(new Range(tInputs.cols()));
		featureSelectionCols.removeAll(featureColumnsToAlwaysUse);
		
		while (!featureSelectionCols.isEmpty())
		{
			// Find the feature selection column which helps the model the most.
			Tuple2<Integer, Double> bestTuple = featureSelectionCols.stream()
					.map(c -> 
					{
						List<Integer> columns = new ArrayList<>(keepColumns);
						columns.add(c);
						SupervisedLearner model = trainOnColumns(tInputs, tLabels, columns);
						double score = validateOnColumns(vInputs, vLabels, columns, model);
						return new Tuple2<Integer, Double>(c, score);
					})
					.max((t1, t2) -> Double.compare(t1.getSecond(), t2.getSecond())).get();

			if (bestTuple.getSecond() > lastIterationScore)
			{
				Logger.println("Keeping column " + bestTuple.getFirst() + ", new score: " + bestTuple.getSecond());
				keepColumns.add(bestTuple.getFirst());
				featureSelectionCols.remove(bestTuple.getFirst());
			}
			else
			{
				break;
			}
			lastIterationScore = bestTuple.getSecond();
		}
		
		return keepColumns;
	}
	
	/**
	 * Trains a model with inputs of all feature columns specified by featuresColumnsToAlwaysUse
	 * plus those in "columns".
	 * @return A tuple with the trained model and a score, which is it's performance on a validation
	 * set.
	 */
	private SupervisedLearner trainOnColumns(Matrix tInputs, Matrix tLabels, List<Integer> columns)
	{
		assert featureColumnsToAlwaysUse.stream().allMatch(c -> columns.contains(c));
		
		SupervisedLearner model = MLSystemsManager.createLearner(rand, innerModelName, innerModelSettings);
		
		// Create the new inputs with only the specified columns.
		Matrix inputs = tInputs.selectColumns(columns);
		
		model.train(inputs, tLabels);

		return model;
	}
	
	private Double validateOnColumns(Matrix vInputs, Matrix vLabels, List<Integer> columns, SupervisedLearner model)
	{	
		// Create the new inputs with only the specified columns.
		Matrix inputs = vInputs.selectColumns(columns);
		
		double score = Evaluator.runEvaluators(inputs, vLabels, model, true, 
				Collections.singletonList(evaluator))
				.getScores(evaluator.getClass()).get(0);
	
		// Make sure that increasing scores are better.
		if (evaluator.higherScoresAreBetter())
			return score;
		return 1.0 - score;
	}
	

	@Override
	public Vector innerPredict(Vector input)
	{
		return innerModel.predict(selectFeatureColumns(input));
	}
	
	private Vector selectFeatureColumns(Vector input)
	{
		ArrayList<Double> result = new ArrayList<>();
		for (int c : innerModelFeatureColumns)
		{
			result.add(input.get(c));
		}
		return new VectorDouble(Helper.toDoubleArray(result), input.getWeight());
	}
	
	@Override
	public List<double[]> innerPredictOutputWeights(Vector input)
	{
		input = selectFeatureColumns(input);
		return innerModel.predictOutputWeights(input);
	}

	/**
	 * Returns the filter from innerModel. This is needed for RankedCC. 
	 * Warning: The resulting filter will only work to filter labels, not inputs.
	 */
	public Filter getInnerModelFilter()
	{
		return innerModel.getFilter();
	}

	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
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
		return false;
	}
	
	public static void testPrivateMethods()
	{
		{
			Matrix data = new Matrix();
			// The second column is irrelevant.
			data.loadFromArffString("\n" + 
					"@RELATION 'test -c -1'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a, b}\n" + 
					"@ATTRIBUTE x2	{c, d}\n" + 
					"@ATTRIBUTE class   {e, f}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"a, c, e\n" + 
					"a, d, e\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f\n" + 
					"b, c, f\n" + 
					"b, d, f");
			
			Matrix inputs = new Matrix(data, 0, 0, data.rows(), data.cols() - data.getNumLabelColumns());
			Matrix labels = new Matrix(data, 0, data.cols() - data.getNumLabelColumns(), data.rows(),
					data.getNumLabelColumns());
	
			FeatureWrapper wrapper = new FeatureWrapper(new ArrayList<Integer>(), "neuralnet", 
					MLSystemsManager.parseModelSettingsFile("model_settings/neuralnet_test.json"), 
					new TopN(Arrays.asList(1)), 0.5, 1);
			wrapper.setRandom(new Random(0));
			wrapper.train(inputs, labels);
			
			assertEquals(Arrays.asList(0), wrapper.innerModelFeatureColumns);
			assertEquals(new VectorDouble(new double[]{1}), wrapper.selectFeatureColumns(new VectorDouble(new double[] {1, 0})));
			assertEquals(new VectorDouble(new double[]{0}), wrapper.selectFeatureColumns(new VectorDouble(new double[] {0, 0})));
		}

		{
			Matrix data = new Matrix();
			// The first column is irrelevant.
			data.loadFromArffString("\n" + 
					"@RELATION 'test -c -1'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{c, d}\n" + 
					"@ATTRIBUTE x2	{a, b}\n" + 
					"@ATTRIBUTE class   {e, f}\n" + 
					"\n" + 
					"@DATA\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, a, e\n" + 
					"d, a, e\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f\n" + 
					"c, b, f\n" + 
					"d, b, f");
			
			Matrix inputs = new Matrix(data, 0, 0, data.rows(), data.cols() - data.getNumLabelColumns());
			Matrix labels = new Matrix(data, 0, data.cols() - data.getNumLabelColumns(), data.rows(),
					data.getNumLabelColumns());
	
			FeatureWrapper wrapper = new FeatureWrapper(new ArrayList<Integer>(), "neuralnet", 
					MLSystemsManager.parseModelSettingsFile("model_settings/neuralnet_test.json"), 
					new TopN(Arrays.asList(1)), 0.5, 1);
			wrapper.setRandom(new Random(0));
			wrapper.train(inputs, labels);
			
			assertEquals(Arrays.asList(1), wrapper.innerModelFeatureColumns);
			assertEquals(new VectorDouble(new double[]{0}), wrapper.selectFeatureColumns(new VectorDouble(new double[] {1, 0})));
			assertEquals(new VectorDouble(new double[]{0}), wrapper.selectFeatureColumns(new VectorDouble(new double[] {0, 0})));
		}

		{
			Matrix data = new Matrix();
			// The class is an AND of the 2 inputs, so both inputs are needed to get predict the class.
			data.loadFromArffString("@RELATION 'test -c -1'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{a0, a1}\n" + 
					"@ATTRIBUTE x2	{b0, b1}\n" + 
					"@ATTRIBUTE class   {c0, c1}\n" + 
					"\n" + 
					"@DATA\n" + 
					"a0,b0,c0\n" + 
					"a0,b1,c0\n" + 
					"a1,b0,c0\n" + 
					"a1,b1,c1\n" + 
					"a0,b0,c0\n" + 
					"a0,b1,c0\n" + 
					"a1,b0,c0\n" + 
					"a1,b1,c1\n" + 
					"a0,b0,c0\n" + 
					"a0,b1,c0\n" + 
					"a1,b0,c0\n" + 
					"a1,b1,c1\n" + 
					"a0,b0,c0\n" + 
					"a0,b1,c0\n" + 
					"a1,b0,c0\n" + 
					"a1,b1,c1\n" + 
					"a0,b0,c0\n" + 
					"a0,b1,c0\n" + 
					"a1,b0,c0\n" + 
					"a1,b1,c1\n" + 
					"a0,b0,c0\n" + 
					"a0,b1,c0\n" + 
					"a1,b0,c0\n" + 
					"a1,b1,c1\n" + 
					"");
			
			Matrix inputs = new Matrix(data, 0, 0, data.rows(), data.cols() - data.getNumLabelColumns());
			Matrix labels = new Matrix(data, 0, data.cols() - data.getNumLabelColumns(), data.rows(),
					data.getNumLabelColumns());
	
			FeatureWrapper wrapper = new FeatureWrapper(new ArrayList<Integer>(), "neuralnet", 
					MLSystemsManager.parseModelSettingsFile("model_settings/neuralnet_test.json"), 
					new TopN(Arrays.asList(1)), 0.5, 1);
			wrapper.setRandom(new Random(0));
			wrapper.train(inputs, labels);
			
			assertEquals(Arrays.asList(0, 1), wrapper.innerModelFeatureColumns);
			assertEquals(new VectorDouble(new double[]{1, 0}), wrapper.selectFeatureColumns(new VectorDouble(new double[] {1, 0})));
			assertEquals(new VectorDouble(new double[]{0, 0}), wrapper.selectFeatureColumns(new VectorDouble(new double[] {0, 0})));
		}
	}

	@Override
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
		return false;
	}
}
