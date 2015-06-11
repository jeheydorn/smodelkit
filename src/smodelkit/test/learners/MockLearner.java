package smodelkit.test.learners;

import java.util.Iterator;
import java.util.List;

import org.json.simple.JSONObject;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.learner.SupervisedLearner;

/**
 * Returns an array of predictions one row at a time when innerPredict is called.
 */
public class MockLearner extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private Iterator<double[]> predIterator;
	
	public MockLearner(List<double[]> predictions)
	{
		this.predIterator = predictions.iterator();
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{
	}

	@Override
	public Vector innerPredict(Vector input)
	{
		return Vector.create(predIterator.next());
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
		return true;
	}

	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}

	@Override
	public void configure(JSONObject settings)
	{
		throw new UnsupportedOperationException();
	}
}