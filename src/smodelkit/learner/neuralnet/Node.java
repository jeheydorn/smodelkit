package smodelkit.learner.neuralnet;

import java.io.Serializable;
import java.util.Random;

import smodelkit.Vector;

/**
 * A node an in aritifial neural network.
 * @author joseph
 *
 */
@SuppressWarnings("serial")
public abstract class Node implements Serializable
{
	private double[] weights;
	protected double momentum;
	
	public Node(Random r, int numInputs, double momentum)
	{
		this.momentum = momentum;
		
		// 1 is added for the bias weight.
		weights = new double[numInputs + 1];

		initializeWeights(r, weights);

	}
	
	protected void initializeWeights(Random rand, double[] weights)
	{
		final double standardDeviation = 0.1;
		for(int i = 0; i < weights.length; i++)
		{
			weights[i] = rand.nextGaussian() * standardDeviation;
			// Below is how weka's MultilayerPerceptron initializes weights.
			//weights[i] = rand.nextDouble() * .1 - .05;
		}
	}
	
	protected double calcNet(Vector inputs)
	{
		double total = 0;
		for(int i = 0; i < inputs.size(); i++)
			total += inputs.get(i) * weights[i];

		// bias weight
		total += weights[weights.length - 1];
		
		return total;
	}

	/**
	 * The squashing function of a network node.
	 */
	public abstract double squash(double net);
	
	public double calcOutput(Vector inputs)
	{
		assert(inputs.size() == weights.length -1);

		double net = calcNet(inputs);
		return squash(net);
	}
	
	public abstract double calcOutputNodeError(double target, double output);
	
	public abstract double calcHiddenNodeError(double errorFromHigherLayer, double output);
	
	public double getWeight(int index)
	{
		assert(index != weights.length -1);
		return weights[index];
	}
	
	public void updateWeights(Vector inputs, double error, double learningRate)
	{
		addWeightChanges(inputs, error, learningRate, weights);
	}
	
	private double[] addWeightChanges(Vector inputs, double error, double learningRate,
			double[] outResult) 
	{
		assert(inputs.size() == weights.length -1);

		if (outResult.length != weights.length)
			throw new IllegalArgumentException();


		for (int i = 0; i < inputs.size(); i++)
		{
			double weightChange = learningRate * error * inputs.get(i) + momentum*weights[i];
			outResult[i] += weightChange;
		}

		// calculate bias weight change
		double weightChange = learningRate * error + momentum*weights[weights.length -1];
		outResult[outResult.length -1] += weightChange;
		return outResult;
	}


	public double[] getWeights() 
	{ 
		return weights; 
	}


}

