package smodelkit.test.learners;

import java.util.Iterator;
import java.util.List;

import org.json.simple.JSONObject;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.learner.SupervisedLearner;
import smodelkit.util.Tuple2;

/**
 * Returns lists of predicted weights one at a time when prdictOutputWeights is called.
 */
public class OutputWeightsMockLearner extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private Iterator<List<double[]>> predictionIter;
	
	public OutputWeightsMockLearner(List<List<double[]>> weightsToPredict)
	{
		this.predictionIter = weightsToPredict.iterator();
	}
	
	@Override
	public List<double[]> innerPredictOutputWeights(Vector input)
	{
		return predictionIter.next();
	}
	
	public List<Tuple2<double[], Double>> innerPredictScoredList(double[] input, int maxDesiredSize)
	{
		throw new UnsupportedOperationException();
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{
	}

	@Override
	public Vector innerPredict(Vector input)
	{
		throw new UnsupportedOperationException();
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
		return false;
	}

	@Override
	public void configure(JSONObject settings)
	{
		throw new UnsupportedOperationException();
	}
}