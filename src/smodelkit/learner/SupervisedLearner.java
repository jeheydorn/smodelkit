package smodelkit.learner;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.json.simple.JSONObject;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.filter.Filter;
import smodelkit.filter.ReorderOutputs;
import smodelkit.util.Bounds;
import smodelkit.util.Range;

/**
 * All learning algorithms should inherit from this class.
 * @author joseph
 *
 */
public abstract class SupervisedLearner implements Serializable
{	
	private static final long serialVersionUID = 1L;
	
	private Filter filter;
	/**
	 * Subclasses can set these ranges if desired.
	 */
	protected Bounds supportedFeatureRange;
	protected Bounds supportedLabelRange;
	
	protected Random rand;

	/**
	 * Stores metadata about the labels.
	 */
	private Matrix labelsMetadata;

	/**
	 * A flag to tell innerPredict and innerPredictScoredList that the default implementation
	 * is being used. This flag is used to prevent infinite recursion if a learner has not
	 * overridden either prediction method.
	 */
	private boolean defaultPredicting;
	
	public SupervisedLearner()
	{
		supportedFeatureRange = new Bounds();
		supportedLabelRange = new Bounds();
	}
		
	public final Filter getFilter()
	{
		return filter;
	}
	
	/**
	 * Applies all filters in the model, then passes the filtered training set to innerTrain().
	 */
	public final void train(Matrix inputs, Matrix labels)
	{
		labelsMetadata = new Matrix();
		labelsMetadata.copyMetadata(labels);
		
		if (filter == null)
		{
			if (!isCompatible(inputs, labels))
				throw new IllegalArgumentException("After applying filters, this model is not compatable with " +
					"the given dataset.");
			innerTrain(inputs, labels);
		}
		else
		{
			initializeFilter(inputs, labels);
			Matrix inputsFiltered = filter.filterAllInputs(inputs);
			Matrix labelsFiltered = filter.filterAllLabels(labels);
			if (!isCompatible(inputsFiltered, labelsFiltered))
				throw new IllegalArgumentException("After applying filters, this model is not compatable with " +
					"the given dataset.");
			innerTrain(inputsFiltered, labelsFiltered);
		}
	}

	/**
	 * This method needs to be exposed so that AccuracyMeasures can be used while training because the filters
	 * are not used while training.
	 * @param useFilter If true, the learning will filter and unfilter the input when predicitng. 
	 */
	public final Vector predict(Vector input, boolean useFilter)
	{
		if (!useFilter || filter == null )
		{
			if (!canImplicitlyHandleUnknownInputs())
				checkForUnknownInputs(input);

			return innerPredict(input);
		}
		else
		{
			Vector filteredInput = filter.filterInput(input);
			
			if (!canImplicitlyHandleUnknownInputs())
				checkForUnknownInputs(filteredInput);
			
			Vector pred = innerPredict(filteredInput);
			Vector predUnfiltered = filter.unfilterLabel(pred);
			return predUnfiltered;
		}
	}
	
	private void checkForUnknownInputs(Vector input)
	{
		// 	Check for unknown values.
		for (int i = 0; i < input.size(); i++)
		{
			if (Vector.isUnknown(input.get(i)))
				throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle unknown inputs.");
		}
	}
	
	/**
	 * Applies all filters in the model to the given input, then passes it to innerPredict. 
	 * The result is then unfiltered.
	 */
	public final Vector predict(Vector input)
	{
		return predict(input, true);
	}
	
	/**
	 * Train the model using the inputs and labels.
	 * @param inputs Filtered inputs for training. This must not be mutated.
	 * @param labels Filtered labels for training. This must not be mutated.
	 */
	protected abstract void innerTrain(Matrix inputs, Matrix labels);

	/**
	 * Predict a label for the given input.
	 * @param input The input to predict a label for. This must not be mutated.
	 */
	protected Vector innerPredict(Vector input)
	{
		if (defaultPredicting)
			throw new IllegalStateException("Learners must override at least one of innerPredict or innerPredictScoredList.");
		defaultPredicting = true;
		Vector result = innerPredictScoredList(input, 1).get(0);
		defaultPredicting = false;
		return result;
	}
	
	/**
	 * Calls predictScoredList(Vector, int, boolean) with useFilter=true.
	 */
	public List<Vector> predictScoredList(Vector input, int maxDesiredSize)
	{
		return predictScoredList(input, maxDesiredSize, true);
	}
	
	/**
	 * For learners that give a scored list of predictions. The score of each prediction should be
	 * stored in the weight field of each predicted Vector.
	 *
	 * The result must be sorted in decreasing order of scores.
	 *
	 *	@param maxDesiredSize The desired size of the result. The purpose of this parameter is to tell
	 * the learner that it doesn't need to create a result larger than this size. The learner
	 * may ignore this parameter.
	 */
	public List<Vector> predictScoredList(Vector input, int maxDesiredSize, boolean useFilter)
	{
		if (filter == null || !useFilter)
		{
			return innerPredictScoredList(input, maxDesiredSize);
		}
		else
		{
			Vector filteredInput = filter.filterInput(input);
			List<Vector> predictions = innerPredictScoredList(filteredInput, maxDesiredSize);
			
			if (predictions.size() == 1)
			{
				// Unfilter the prediction. 
				return Collections.singletonList(new Vector(
						filter.unfilterLabel(predictions.get(0))));
			}
			
			// Unfilter the predictions.
			List<Vector> unfiltered = new ArrayList<>(predictions.size());
			for (Vector pred : predictions)
			{
				unfiltered.add(filter.unfilterLabel(pred));
			}
			return unfiltered;
			
		}
	}
	
	/**
	 * Override this for it to do something meaningful. 
	 * 
	 * For learners that give a scored list of predictions. The score of each prediction should be
	 * stored in the weight field of each predicted Vector.

	 * The list must be sorted in decreasing order of scores.
	 * 
	 * @param input The input for which a prediction is to be made. Values in this array must not be mutated.
	 * @param maxDesiredSize See predictSocredList.
	 */
	protected List<Vector> innerPredictScoredList(Vector input, int maxDesiredSize)
	{
		if (defaultPredicting)
			throw new IllegalStateException("Learners must override at least one of innerPredict or innerPredictScoredList.");
		defaultPredicting = true;
		Vector pred = innerPredict(input);
		pred.setWeight(1.0);
		List<Vector> result = Collections.singletonList(pred);
		defaultPredicting = false;
		return result;
	}
	
	/**
	 * Get the weights associated with each nominal value in a predicted output
	 * vector.
	 * 
	 * Models which use this method should implement innerGetOutputWeights.
	 * 
	 * @param input The input vector to make a prediction for. Values in this array must not be mutated.
	 * @return Each element in the result corresponds to an output dimension. For single
	 * dimensional outputs, the result will be a list with only 1 double array. The
	 * value in each double array are the predicted weights corresponding to each possible
	 * output value for a dimension. 
	 */
	public final List<double[]> predictOutputWeights(Vector input)
	{
		if (filter == null)
		{
			if (!canImplicitlyHandleUnknownInputs())
				checkForUnknownInputs(input);

			return innerPredictOutputWeights(input);
		}
		else
		{
			// If NominalToCategorical or Normalize were used in filter, then I should not apply them
			// to weights. But if ReorderOutputs was used, I must reverse the order of the weight
			// vectors here.
			
			Vector inputFiltered = filter.filterInput(input);
			
			if (!canImplicitlyHandleUnknownInputs())
				checkForUnknownInputs(inputFiltered);			
			
			List<double[]> weights = innerPredictOutputWeights(inputFiltered);
			ReorderOutputs reorder = filter.findFilter(ReorderOutputs.class);
			if (reorder == null)
				return weights;
			else
				return reorder.unfilterOutputWeights(weights);	
		}
	}

	/**
	 * Override this to support this functionality. The default is to put all weight on
	 * the values predicted by predict().
	 * @param input The input for which a prediction is to be made. Values in this array must not be mutated.
	 * @return See predictOutputWeights.
	 */
	protected List<double[]> innerPredictOutputWeights(Vector input)
	{
		Vector pred = predict(input);
		List<double[]> weights = new ArrayList<>();
		for (int c : new Range(pred.size()))
		{
			if (labelsMetadata.isContinuous(c))
			{
				weights.add(new double[]{pred.get(c)});
			}
			else
			{
				double[] w = new double[labelsMetadata.getValueCount(c)];
				w[(int)pred.get(c)] = 1.0;
				weights.add(w);
			}
		}
		
		return weights;
	}

		
	/**
	 * Checks if the given dataset is compatible with this learner.
	 */
	public final boolean isCompatible(Matrix inputs, Matrix labels)
	{
		if (inputs.hasNominalCols() && !canImplicitlyHandleNominalFeatures())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle nominal features.");
				
		if (inputs.hasContinuousCols() && !canImplicitlyHandleContinuousFeatures())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle continuous features.");
		
		if (labels.hasNominalCols() && !canImplicitlyHandleNominalLabels())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle nominal labels.");
		
		if (labels.hasContinuousCols() && !canImplicitlyHandleContinuousLabels())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle continuous labels.");
		
		if (inputs.containsUnknowns() && !canImplicitlyHandleUnknownInputs())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle unknown inputs.");

		if (labels.containsUnknowns() && !canImplicitlyHandleUnknownOutputs())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle unknown outputs.");

		if (labels.cols() > 1 && !canImplicitlyHandleMultipleOutputs())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle multiple outputs.");
		
		if (inputs.hasInstanceWeightsNot1() && !canImplicitlyHandleInstanceWeights())
			throw new IllegalArgumentException(this.getClass().getSimpleName() + " cannot handle weighted instances");
		
		for (int col = 0; col < inputs.cols(); col++)
		{
			if (!inputs.isContinuous(col))
				continue;
			if (supportedFeatureRange.upper < inputs.findMax(col))
				throw new IllegalArgumentException(
						String.format("The highest value of column %s of the given features is beyond the" +
								" upper range of this learner. Column max=%s, learner upper range=%s.", 
								col, inputs.findMax(col), supportedFeatureRange.upper));
			if (supportedFeatureRange.lower > inputs.findMin(col))
			{
				throw new IllegalArgumentException(
						String.format("The lowest value of column %s of the given features is below the" +
								" lower range of this learner. Column min=%s, learner lower range=%s.", 
								col, inputs.findMin(col), supportedFeatureRange.lower));
			}
		}
		
		for (int col = 0; col < labels.cols(); col++)
		{
			if (!labels.isContinuous(col))
				continue;
			if (supportedLabelRange.upper < labels.findMax(col))
				throw new IllegalArgumentException(
						String.format("The highest value of column %s of the given labels is beyond the" +
								" upper range of this learner. Column max=%s, learner upper range=%s.", 
								col, labels.findMax(col), supportedLabelRange.upper));
			if (supportedLabelRange.lower > labels.findMin(col))
				throw new IllegalArgumentException(
						String.format("The lowest value of column %s of the given labels is below the" +
						" lower range of this learner. Column min=%s, learner lower range=%s.", 
						col, labels.findMin(col), supportedLabelRange.lower));
		}

		
		return true;
	}
	
	protected abstract boolean canImplicitlyHandleNominalFeatures();
	
	protected abstract boolean canImplicitlyHandleContinuousFeatures();
	
	protected abstract boolean canImplicitlyHandleNominalLabels();
	
	protected abstract boolean canImplicitlyHandleContinuousLabels();
	
	protected abstract boolean canImplicitlyHandleUnknownInputs();
	
	protected abstract boolean canImplicitlyHandleUnknownOutputs();
	
	protected abstract boolean canImplicitlyHandleMultipleOutputs();
	
	public abstract boolean canImplicitlyHandleInstanceWeights();

	/**
	 * Splits a validation set off of the training set.
	 * @return An array of datasets where:
	 * 0 - training inputs
	 * 1 - training labels
	 * 2 - validation inputs
	 * 3 - validation labels
	 */
	protected final Matrix[] createValidationSet(Matrix inputs, Matrix labels,
			double validationSetPercent)
	{
		Matrix[] result = new Matrix[4];
		
		if (validationSetPercent == 0 || validationSetPercent == 1)
		{
			// Validate using training set.
			result[0] = inputs;
			result[1] = labels;
			result[2] = inputs;
			result[3] = labels;
			return result;
		}

		int validationSetSize = (int) (inputs.rows() * validationSetPercent);

		result[0] = new Matrix();
		result[1] = new Matrix();
		result[2] = new Matrix();
		result[3] = new Matrix();

		result[0].copyMetadata(inputs);
		result[1].copyMetadata(labels);
		result[2].copyMetadata(inputs);
		result[3].copyMetadata(labels);

		result[0].addRows(inputs, validationSetSize, inputs.rows()
				- validationSetSize);
		result[1].addRows(labels, validationSetSize, labels.rows()
				- validationSetSize);
		result[2].addRows(inputs, validationSetSize);
		result[3].addRows(labels, validationSetSize);
		assert (result[0].rows() == inputs.rows() - validationSetSize);
		assert (result[0].cols() == result[2].cols());
		
		return result;
	}
	
	private final void initializeFilter(Matrix inputs, Matrix labels)
	{
		if (filter != null)
			filter.initialize(inputs, labels);
	}

	public void setFilter(Filter newFilter)
	{
		filter = newFilter;
	}
	
	/**
	 * Configures a learner from settings stored in json.
	 */
	public abstract void configure(JSONObject settings);
	
	/**
	 * Sets the random number generator.
	 * @param r
	 */
	public void setRandom(Random r)
	{
		this.rand = r;
	}
	
	/**
	 * Checks if the specified argument is in the key of given settings. If not, an exception
	 * is thrown.
	 */
	public void checkNullableArgumentIsPresent(JSONObject settings, String argumentName)
	{
		if (!settings.containsKey(argumentName))
			throw new IllegalArgumentException("Missing argument \"" + argumentName + "\". If you do not wish to use this argument, "
					+ "set it to null in the json settings file.");

	}
}
