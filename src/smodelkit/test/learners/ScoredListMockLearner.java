package smodelkit.test.learners;

import java.util.Iterator;
import java.util.List;

import org.json.simple.JSONObject;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.learner.SupervisedLearner;

/**
 * Returns lists of predicted output vectors one at a time when predictScoredList is called.
 */
public class ScoredListMockLearner extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private Iterator<List<Vector>> predictionIter;
	
	public ScoredListMockLearner(List<List<Vector>> predictions)
	{
		this.predictionIter = predictions.iterator();
	}
	
	@Override
	public List<Vector> innerPredictScoredList(Vector input, int maxDesiredSize)
	{
		return predictionIter.next();
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
	public List<double[]> innerPredictOutputWeights(
			Vector input)
	{
		throw new UnsupportedOperationException();
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