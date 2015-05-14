package smodelkit.learner;
import java.io.Serializable;
import java.util.Random;

import smodelkit.Vector;
//import static java.lang.System.out;


public class SigmoidNode implements Serializable
{
	private static final long serialVersionUID = 1L;
	private final double DEFAULT_WEIGHT = 0.0;
	private boolean USE_RANDOM_WEIGHTS = true;
	private double RANDOM_WEIGHT_STANDARD_DEVIATION = 0.1;
	
	private double[] weights;
	protected double momentum;
	
	public SigmoidNode(Random r, int numInputs, double momentum)
	{
		this.momentum = momentum;
		
		// 1 is added for the bias weight.
		weights = new double[numInputs + 1];

		initializeWeights(r, weights);
	}
	
	protected void initializeWeights(Random rand, double[] weights)
	{
		for(int i = 0; i < weights.length; i++)
		{
			if (USE_RANDOM_WEIGHTS)
			{
				weights[i] = rand.nextGaussian()*RANDOM_WEIGHT_STANDARD_DEVIATION;
				// Below is how weka's MultilayerPerceptron initializes weights.
				//weights[i] = rand.nextDouble() * .1 - .05;
			}
			else
			{
				weights[i] = DEFAULT_WEIGHT;
			}
		}
	}

	double calcNet(Vector inputs)
	{
		double total = 0;
		for(int i = 0; i < inputs.size(); i++)
			total += inputs.get(i) * weights[i];

		// bias weight
		total += weights[weights.length - 1];
		
		return total;
	}

	double calcOutput(Vector inputs)
	{
		assert(inputs.size() == weights.length -1);

		double net = calcNet(inputs);
		return sig(net);
	}
	
	public static double sig(double net)
	{
		return 1/(1 + Math.exp(-net));
	}
	
	double getWeight(int index)
	{
		assert(index != weights.length -1);
		return weights[index];
	}
	
	void updateWeights(Vector inputs, double error, double learningRate, double weightDecayRate)
	{
		addWeightChanges(inputs, error, learningRate, weightDecayRate, weights);
	}
	
	private double[] addWeightChanges(Vector inputs, double error, double learningRate,
			double weightDecayRate, double[] outResult) 
	{
		assert(inputs.size() == weights.length -1);

		if (outResult.length != weights.length)
			throw new IllegalArgumentException();


		for (int i = 0; i < inputs.size(); i++)
		{
			double weightChange = learningRate * error * inputs.get(i) + momentum*weights[i] 
					- learningRate * weightDecayRate * weights[i];
			outResult[i] += weightChange;
		}

		// calculate bias weight change
		double weightChange = learningRate * error + momentum*weights[weights.length -1] 
				- learningRate * weightDecayRate * weights[weights.length -1];
		outResult[outResult.length -1] += weightChange;
		return outResult;
	}


	double[] getWeights() 
	{ 
		return weights; 
	}


}
