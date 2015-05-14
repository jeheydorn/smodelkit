package smodelkit.evaluator;

import java.io.Serializable;
import java.util.List;

import smodelkit.ConfusionMatrix;
import smodelkit.Evaluation;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.learner.SupervisedLearner;
import smodelkit.util.Range;

/**
 * Evaluators are used to give a score to predictions from a learner.
 * 
 * Remember when implementing an evaluator to cast a target value or prediction to 
 * an int without first checking if it is an unknown value using Matrix.isUnknown().
 * This is because (int)Double.NaN == 0.
 * @author joseph
 *
 */
public abstract class Evaluator implements Serializable
{
	private static final long serialVersionUID = 1L;

	public Evaluator()
	{
	}

	/**
	 * Returns true if increasing scores mean a better evaluation. Otherwise this must return false.
	 * Evaluators which measure error should return false.
	 * @return
	 */
	public abstract boolean higherScoresAreBetter();
		
	/**
	 * Configure this evaluator from arguments that were with it's name in a settings file.
	 * @param args
	 */
	public abstract void configure(String[] args);
	
	/**
	 * This tells the evaluator to reset its internal state to begin a new batch.
	 * @param metadata A matrix with meta data for the labels that will be evaluated.
	 */
	protected abstract void startBatch(Matrix metadata);
	
	/**
	 * This updates the evaluator for the given target and prediction.
	 * @param target The target output vector.
	 * @param predictions A predicted list of output vectors from the learner being evaluated. This
	 * vectors must not be mutated by the evaluator.
	 */
	protected abstract void evaluate(Vector target, List<Vector> predictions);
	
	/**
	 * Calculates the current score based on the evaluations that have been made so far.
	 * @return
	 */
	protected abstract List<Double> calcScores();
	
	/**
	 * Calculates confusions matrixes for the evaluations that have been done so far.
	 * @return This may be null if this evaluator does not support confusion matrices.
	 */
	protected abstract List<ConfusionMatrix> calcConfusions();
	
	/**
	 * Returns the maximum number of output vectors a learner should predict when 
	 * calling SupervisedLearner.predictScoredList. Since there may be an intractably
	 * large number of possible output vectors (all possible combinations of output values),
	 * it is important that this be a reasonable value.
	 * 
	 * The learner does not have to restrict its predictions to this number.
	 */
	protected abstract int getMaxDesiredSize();
	
	/**
	 * Evaluates the given learner on the given dataset using the given evaluators.
	 * 
	 * @param maxDesiredSize See maxDesiredSize in SupervisedLearner.predictScoredList.
	 * @param learnerUseFilter This tells the learner that it needs to apply its filter
	 * to the given inputs and labels when making predictions. In this case the filter will also
	 * be used in reverse to unfilter the learner's predictions before passing them to the 
	 * evaluator. The only reason this should be false is if the inputs and labels are
	 * pre-filterd, such as in SupervisedLearner.innerTrain.
	 */
	public static Evaluation runEvaluators(Matrix inputs, Matrix labels, SupervisedLearner learner, 
			boolean learnerUseFilter, List<Evaluator> evaluators)
	{
		if (inputs.rows() != labels.rows())
			throw new IllegalArgumentException();
		
		evaluators.stream().forEach(evaluator -> evaluator.startBatch(labels));
		int maxDesiredSize = evaluators.stream().mapToInt(
				evaluator -> evaluator.getMaxDesiredSize()).max().getAsInt();
		
		for (int r : new Range(inputs.rows()))
		{
			List<Vector> predictions = learner.predictScoredList(inputs.row(r), 
					maxDesiredSize, learnerUseFilter);
			for (Evaluator evaluator : evaluators)
			{
				evaluator.evaluate(labels.row(r), predictions);
			}
		}
		
		Evaluation evaluation = new Evaluation();
		for (Evaluator evaluator : evaluators)
		{
			evaluation.putScores(evaluator.getClass(), evaluator.calcScores());
			evaluation.putConfusions(evaluator.getClass(), evaluator.calcConfusions());
		}
		
		return evaluation;
		
	}

}

