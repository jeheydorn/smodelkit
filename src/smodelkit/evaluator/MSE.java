package smodelkit.evaluator;

import java.util.Arrays;
import java.util.List;

import smodelkit.ConfusionMatrix;
import smodelkit.Matrix;
import smodelkit.Vector;

/**
 * Evaluates a SupervisedLearner on a dataset. Finds mean sum squared error.
 * @author joseph
 *
 */
public class MSE extends Evaluator
{
	private static final long serialVersionUID = 1L;
	private double sse;
	private double totalWeight;
	
	public MSE()
	{
	}

	@Override
	public void configure(String[] args)
	{
	}

	@Override
	protected void startBatch(Matrix metadata)
	{
		sse = 0.0;
		totalWeight = 0;
		
	}

	@Override
	protected void evaluate(Vector target, List<Vector> predictions)
	{
		Vector prediction = predictions.get(0);
		// Weight the error of each prediction according to the weight of that instance.
		sse += calcMSE(target, prediction) * target.getWeight();
		totalWeight += target.getWeight();
	}

	@Override
	protected List<Double> calcScores()
	{
		double result = sse / totalWeight;
		return Arrays.asList(result);
	}

	@Override
	protected List<ConfusionMatrix> calcConfusions()
	{
		return null;
	}

	@Override
	protected int getMaxDesiredSize()
	{
		return 1;
	}
	
	@Override
	public boolean higherScoresAreBetter()
	{
		return false;
	}

	private double calcMSE(Vector expected, Vector actual)
	{
		assert expected.size() == actual.size();
		
		double sse = 0.0;
		for (int lCol = 0; lCol < expected.size(); lCol++)
		{
			double delta = expected.get(lCol) - actual.get(lCol);
			sse += (delta * delta);
		}
		return sse / expected.size();
	}



}
