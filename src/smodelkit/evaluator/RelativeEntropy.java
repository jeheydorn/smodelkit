package smodelkit.evaluator;

import java.util.Arrays;
import java.util.List;

import smodelkit.ConfusionMatrix;
import smodelkit.Matrix;
import smodelkit.Vector;

/**
 * Find the average relative entropy of targets and predictions.
 * 
 *  This is based on equation 2.15 from "Accelerated Learning in Layered Neural Networks" 
 *  by Sara A. Solla, Esther Levin, and Michael Fleisher.
 *  
 * @author joseph
 *
 */
public class RelativeEntropy extends Evaluator
{
	private static final long serialVersionUID = 1L;
	private double errorSum;
	private double totalWeight;
	
	public RelativeEntropy()
	{
	}

	@Override
	public void configure(String[] args)
	{
	}

	@Override
	protected void startBatch(Matrix metadata)
	{
		errorSum = 0.0;
		totalWeight = 0;
		
	}

	@Override
	protected void evaluate(Vector target, List<Vector> predictions)
	{
		Vector prediction = predictions.get(0);
		// Weight the error of each prediction according to the weight of that instance.
		errorSum += calcRelativeEntropy(target, prediction) * target.getWeight();
		totalWeight += target.getWeight();
	}

	private double calcRelativeEntropy(Vector target, Vector prediction)
	{
		assert target.size() == prediction.size();
		
		target = softMax(target);
		prediction = softMax(prediction);
		
		double sum = 0;
		for (int i = 0; i < target.size(); i++)
		{
			sum += target.get(i) * Math.log(target.get(i)/prediction.get(i)) + 
					(1 - target.get(i)) * Math.log((1 - target.get(i))/(1 - prediction.get(i)));
		}
		return sum;
	}
	
	/**
	 * Computes the softmax function.
	 */
	private Vector softMax(Vector v)
	{
		double[] values = new double[v.size()];
		for (int i = 0; i < v.size(); i++)
		{
			values[i] = v.get(i);
		}
		softMaxInPlace(values);
		return new Vector(values, v.getWeight());
	}
	
	public static void softMaxInPlace(double[] values)
	{
		double total = 0;
		for (int i = 0; i < values.length; i++)
		{
			values[i] = Math.exp(values[i]);
			total += values[i];
		}
		for (int i = 0; i < values.length; i++)
		{
			values[i] /= total;
		}
	}

	@Override
	protected List<Double> calcScores()
	{
		double result = errorSum / totalWeight;
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




}
