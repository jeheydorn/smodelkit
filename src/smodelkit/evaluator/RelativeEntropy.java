package smodelkit.evaluator;

import java.util.Arrays;
import java.util.List;

import smodelkit.ConfusionMatrix;
import smodelkit.Matrix;
import smodelkit.Vector;

/**
 * Find the average relative entropy of targets and predictions.
 * 
 * This is based on "Cost Function" in 
 * http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression#Relationship_to_Logistic_Regression
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
		
		prediction = softmax(prediction);
		
		double sum = 0;
		for (int i = 0; i < target.size(); i++)
		{
			if (target.get(i) != 0)
			{
				sum += target.get(i) * Math.log(prediction.get(i));
			}
		}
		return -sum;
	}
	
	/**
	 * Computes the softmax function.
	 */
	public static Vector softmax(Vector v)
	{
		double[] values = new double[v.size()];
		for (int i = 0; i < v.size(); i++)
		{
			values[i] = v.get(i);
		}
		softmaxInPlace(values);
		return new Vector(values, v.getWeight());
	}
	
	public static void softmaxInPlace(double[] values)
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
